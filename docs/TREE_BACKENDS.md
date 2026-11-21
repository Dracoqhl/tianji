# 风电树模型后端

`scripts/tree_model.py` 提供统一的 `train_tree`、`predict_tree`、`save_tree_model`、`load_tree_model` 和 `best_iteration` 接口。XGBoost 使用原生 DMatrix，按保存的名称和类别 schema 训练与预测。T1/T3 的 Huber 配置映射为 pseudo-Huber，T2 的对数电量回归映射为平方误差。原训练样本权重进入训练矩阵，验证矩阵保持无权重。XGBoost 的最佳轮次从零基转成树数，固定轮次重训使用全部树。 [XGBoost 3.0.5 Python API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html) · [类别特征](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/categorical.html)

暖季 T1 保留原同伴 LightGBM CPU 训练和导出路径，GPU 路径复用同样的特征缓存、特征顺序、训练掩码和样本权重。A/B 保留已登记种子与日期，后端写入运行配置、收据和冻结记录。训练打印阶段、首轮、每 25 轮及最后预算轮次。GPU 运行默认将数据处理限制到 4 个线程。CUDA 请求会检查 XGBoost 构建与 booster 的实际设备，避免 CPU 回退。

## 小样本验收

使用服务器已配置的 `/workspace/ctt/model_augmentation/final_check/.venv/bin/python`。先运行 50,000 × 24 的 CUDA 合成测试，包括实际 booster 设备、显存进程快照和保存加载一致性。接着对暖季 T1 入口和独立 T2 特征各做 20 个训练日、10 个 A 集日、最多 20 棵树的 CPU LightGBM 与 GPU XGBoost 对照。T2 小样以训练末段拟合乘法因子，用同一后端训练校准模型和主模型，预测冻结后再打开 A 集评分标签。现有正式 T2 校准与最终候选入口位于服务器项目，接入需要对应源码。小样结果归各自后端与窗口。

原子命令通过固定 GitHub 提交分发，服务器使用已配置环境。运行结果包含模型 schema、分数、耗时和日志。准备完整暖季 T1 GPU 对照时，在同一固定提交下依次使用以下命令。末阶段沿用 A 集冻结记录和六份 B 集预测校验。

```bash
python=/workspace/ctt/model_augmentation/final_check/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
"$python" -u scripts/experiment_companion_t1.py a --backend xgboost --device cuda --threads 4
"$python" -u scripts/experiment_companion_t1.py b-predict --backend xgboost --device cuda --threads 4
"$python" -u scripts/experiment_companion_t1.py b-score --backend xgboost --device cuda --threads 4
```

`scripts/check_tree_submission.py --t1 <T1 CSV> --t2 <T2 CSV> --t3 <T3 CSV>` 对照赛事模板核对列和键的精确顺序、行数、逐站日点数、有限值和物理范围。小样和正式比赛候选文件分别留存。
