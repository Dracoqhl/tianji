"""Shared tree training, inference and persistence for the wind-power experiments."""

from dataclasses import dataclass
import json
from time import monotonic
import warnings

import numpy as np


OBJECTIVES = {'huber': 'reg:pseudohubererror', 'regression': 'reg:squarederror',
              'regression_l2': 'reg:squarederror', 'l2': 'reg:squarederror'}
LOG_PERIOD = 25


def check_backend(backend, device):
    if backend == 'lightgbm':
        if device != 'cpu':
            raise ValueError('The LightGBM backend supports only --device cpu. Use xgboost for CUDA.')
        import lightgbm

        return lightgbm.__version__
    if backend != 'xgboost' or device not in ('cpu', 'cuda'):
        raise ValueError(f'Unsupported tree backend/device: {backend}/{device}.')
    import xgboost

    if device == 'cuda' and not xgboost.build_info().get('USE_CUDA', False):
        raise RuntimeError('This XGBoost build has no CUDA support. Use the server GPU environment.')
    return xgboost.__version__


def xgboost_parameters(params, device, *, custom_metric=False):
    objective = params.get('objective', 'huber')
    if objective not in OBJECTIVES:
        raise ValueError(f'Unsupported tree objective: {objective!r}.')
    config = {
        'tree_method': 'hist', 'device': device, 'grow_policy': 'lossguide',
        'max_bin': 256, 'max_cat_to_onehot': 32, 'max_depth': params.get('max_depth', 0),
        'objective': OBJECTIVES[objective], 'eval_metric': 'rmse',
        'eta': params.get('learning_rate', 0.05), 'max_leaves': params.get('num_leaves', 31),
        'min_child_weight': params.get('min_child_samples', params.get('min_data_in_leaf', 20)),
        'subsample': params.get('subsample', 1.0), 'colsample_bytree': params.get('colsample_bytree', 1.0),
        'alpha': params.get('reg_alpha', 0.0), 'lambda': params.get('reg_lambda', 0.0),
        'seed': params.get('random_state', params.get('seed', 42)),
        'nthread': params.get('n_jobs', params.get('num_threads', 4)),
    }
    if config['max_depth'] <= 0:
        config['max_depth'] = 0
    if custom_metric:
        config.pop('eval_metric')
        config['disable_default_eval_metric'] = True
    if objective == 'huber' and 'huber_slope' in params:
        config['huber_slope'] = params['huber_slope']
    return config


def matrix_shape(X, names):
    if hasattr(X, 'columns') and list(X.columns) != list(names):
        raise ValueError('Feature columns do not match the saved names and order.')
    if len(X.shape) != 2 or X.shape[1] != len(names):
        raise ValueError(f'Expected a feature matrix with {len(names)} columns.')
    return X.shape[0]


def make_dmatrix(xgb, X, names, types, nthread, *, label=None, weight=None):
    matrix_shape(X, names)
    if isinstance(X, np.ndarray):
        for index, feature_type in enumerate(types):
            if feature_type == 'c':
                codes = X[:, index]
                codes = codes[~np.isnan(codes)]
                if not (np.isfinite(codes) & (codes >= 0) & (codes == np.floor(codes))).all():
                    raise ValueError(f'{names[index]} must use nonnegative integer category codes.')
    return xgb.DMatrix(X, label=label, weight=weight, feature_names=list(names), feature_types=list(types),
                       enable_categorical=True, nthread=nthread)


def fitted_device(booster):
    return json.loads(booster.save_config())['learner']['generic_param']['device']


def require_device(booster, device):
    actual = fitted_device(booster)
    if actual.split(':', 1)[0] != device:
        raise RuntimeError(f'XGBoost requested {device} but used {actual}.')
    return actual


def reject_cuda_fallback():
    warnings.filterwarnings('error', message='(?is).*(no visible gpu|setting device to cpu|not compiled with cuda).*',
                            category=UserWarning)


@dataclass
class XGBModel:
    booster: object
    feature_names: tuple
    feature_types: tuple
    nthread: int
    device: str

    @property
    def best_iteration(self):
        value = self.booster.attr('best_iteration')
        return None if value is None else int(value)

    def predict(self, X, iteration_range=None):
        import xgboost as xgb

        matrix = make_dmatrix(xgb, X, self.feature_names, self.feature_types, self.nthread)
        if iteration_range is None:
            best = self.best_iteration
            iteration_range = (0, 0) if best is None else (0, best + 1)
        with warnings.catch_warnings():
            if self.device == 'cuda':
                reject_cuda_fallback()
            prediction = self.booster.predict(matrix, iteration_range=iteration_range)
        require_device(self.booster, self.device)
        return prediction


def print_progress(round_number, rounds, started, scores=()):
    details = ''.join(f' {name}={value:.6g}' for name, value in scores)
    print(f'  trees={round_number}/{rounds}{details} elapsed={monotonic() - started:.1f}s', flush=True)


def lightgbm_progress(period=LOG_PERIOD):
    started = monotonic()

    def report(env):
        step = env.iteration - env.begin_iteration + 1
        rounds = env.end_iteration - env.begin_iteration
        if step == 1 or step % period == 0 or step == rounds:
            scores = [(f'{entry[0]}/{entry[1]}', entry[2]) for entry in env.evaluation_result_list or []]
            print_progress(step, rounds, started, scores)

    report.order = 20  # Report before the early-stopping callback can end the round.
    report.before_iteration = False
    return report


def train_tree(X_train, y_train, feature_names, params, *, X_valid=None, y_valid=None,
               categorical_features=(), early_stopping_rounds=100, sample_weight=None,
               backend='lightgbm', device='cpu', evaluation=None, maximize=None, valid_weight=None,
               log_period=LOG_PERIOD):
    """Fit a native tree model. XGBoost validation predictions use the best boosting round."""
    check_backend(backend, device)
    names = tuple(feature_names)
    categories = set(categorical_features)
    if len(names) != len(set(names)) or not categories.issubset(names):
        raise ValueError('Feature names must be unique and contain every categorical column.')
    ntrain = matrix_shape(X_train, names)
    rounds = int(params['n_estimators'])
    nthread = int(params.get('n_jobs', params.get('num_threads', 4)))
    if ntrain == 0 or rounds < 1 or nthread < 1 or log_period < 0:
        raise ValueError('Training rows, rounds and threads must be positive; log period cannot be negative.')
    if len(y_train) != ntrain or (sample_weight is not None and len(sample_weight) != ntrain):
        raise ValueError('Training labels and weights must match the training rows.')
    if (X_valid is None) != (y_valid is None):
        raise ValueError('Pass validation features and labels together.')
    if X_valid is None and valid_weight is not None:
        raise ValueError('Validation weights require validation data.')
    if backend == 'xgboost' and valid_weight is not None:
        raise ValueError('XGBoost validation is unweighted in this workflow.')
    if X_valid is not None:
        nvalid = matrix_shape(X_valid, names)
        if nvalid == 0 or len(y_valid) != nvalid or (valid_weight is not None and len(valid_weight) != nvalid):
            raise ValueError('Validation labels and weights must match nonempty validation rows.')
    stopping = early_stopping_rounds if X_valid is not None and early_stopping_rounds else None
    if stopping is not None and stopping < 1:
        raise ValueError('Early stopping must use a positive patience.')

    if backend == 'lightgbm':
        import lightgbm as lgb

        config = {key: value for key, value in params.items() if key not in ('n_estimators', 'n_jobs', 'random_state')}
        config.update(num_threads=nthread, seed=params.get('random_state', params.get('seed', 42)))
        config.setdefault('objective', 'huber')
        config.setdefault('metric', 'None' if evaluation else 'l2')
        types = list(categorical_features)
        training = lgb.Dataset(X_train, label=y_train, weight=sample_weight,
                               feature_name=list(names), categorical_feature=types)
        valid = None if X_valid is None else lgb.Dataset(
            X_valid, label=y_valid, weight=valid_weight, feature_name=list(names),
            categorical_feature=types, reference=training,
        )
        callbacks = [lightgbm_progress(log_period)] if log_period else []
        if stopping:
            callbacks.append(lgb.early_stopping(stopping, verbose=bool(log_period)))

        def lightgbm_metric(raw, dataset):
            name, value = evaluation(raw, dataset)
            return name, value, bool(maximize)

        return lgb.train(config, training, num_boost_round=rounds, valid_sets=None if valid is None else [valid],
                         valid_names=None if valid is None else ['validation'],
                         feval=lightgbm_metric if evaluation else None, callbacks=callbacks)

    import xgboost as xgb

    config = xgboost_parameters(params, device, custom_metric=evaluation is not None)
    types = tuple('c' if name in categories else 'q' for name in names)
    training = make_dmatrix(xgb, X_train, names, types, nthread, label=y_train, weight=sample_weight)
    valid = None if X_valid is None else make_dmatrix(xgb, X_valid, names, types, nthread, label=y_valid)
    started = monotonic()

    class Progress(xgb.callback.TrainingCallback):
        def before_training(self, model):
            require_device(model, device)
            return model

        def after_iteration(self, model, epoch, evals_log):
            step = epoch + 1
            if log_period and (step == 1 or step % log_period == 0 or step == rounds):
                scores = [(f'{dataset}/{metric}', values[-1]) for dataset, metrics in evals_log.items()
                          for metric, values in metrics.items()]
                print_progress(step, rounds, started, scores)
            return False

    with warnings.catch_warnings():
        if device == 'cuda':
            reject_cuda_fallback()
        booster = xgb.train(
            config, training, num_boost_round=rounds, evals=[] if valid is None else [(valid, 'validation')],
            early_stopping_rounds=stopping, verbose_eval=False, custom_metric=evaluation, maximize=maximize,
            callbacks=[Progress()],
        )
    require_device(booster, device)
    return XGBModel(booster, names, types, nthread, device)


def best_iteration(model, *, backend, fallback):
    if backend == 'xgboost':
        best = model.best_iteration
        return model.booster.num_boosted_rounds() if best is None else best + 1
    return int(getattr(model, 'best_iteration_', 0) or getattr(model, 'best_iteration', 0) or fallback)


def predict_tree(model, X, *, backend, threads=None):
    if backend == 'xgboost':
        return model.predict(X)
    return model.predict(X, **({'num_threads': threads} if threads is not None else {}))


def save_tree_model(model, path):
    import joblib

    joblib.dump(model, path)


def load_tree_model(path):
    import joblib

    model = joblib.load(path)
    if isinstance(model, XGBModel):
        check_backend('xgboost', model.device)
        with warnings.catch_warnings():
            if model.device == 'cuda':
                reject_cuda_fallback()
            model.booster.set_param({'device': model.device, 'nthread': model.nthread})
        require_device(model.booster, model.device)
    return model
