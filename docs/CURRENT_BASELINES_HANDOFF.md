# T1 v4、T2 v3、T3 v3 服务器交接

服务器源仓库为 `https://github.com/Dracoqhl/tianji.git`。仓库根的 `scripts/` 是代码路径。历史批次通过 Git 解包到临时目录，日常可直接读取的产物在 `/workspace/qhl/runs/`。本地工程的三份配置是 `next-competition/configs/t1_current_baseline_v4.json`、`t2_current_baseline_v3.json`、`t3_current_baseline_v3.json`。

## 先核对产物并复评

使用本交接版本的 `scripts/replay_current_baselines.py`。该入口核对历史预测锁，逐种子重建现行融合并与存档比较，然后生成新的集成 CSV。预测阶段只读既有预测和存档回执。计分阶段先校验所有输出再读取评估标签。用比赛服务器既有 Python 环境，复核阶段只使用 CPU。

```bash
python=/workspace/ctt/model_augmentation/final_check/.venv/bin/python
out=/workspace/qhl/runs/current_baselines_replay_v1
"$python" scripts/replay_current_baselines.py predict --phase B --out "$out"
"$python" scripts/replay_current_baselines.py score --phase B --out "$out"
```

运行结果在 `$out/B/scores.json`。每轨的三个种子和集成 CSV 在 `$out/B/T1/B/`、`$out/B/T2/B/`、`$out/B/T3/fixed_50_10/`。校验的三种子为 `20261020`、`20261037`、`20261117`，逐键等权平均预测后再计分。`--phase A` 可复核 T1/T2 的 A 和 T3 的两个早窗，使用种子 `20261020`。单轨复核可加 `--tracks T1` 等并另给 `--out`，两阶段使用同一组参数。

| 赛道 | B 或主窗口 | 待核对的本地集成主分 | 集成 CSV 键数 |
|---|---|---:|---:|
| T1 v4 | 时序 630 日训练，90 日 B | 88.5937728235 | 103,680 |
| T2 v3 | 时序 630 日训练，90 日 B | 67.5741989772 | 1,080 |
| T3 v3 | 50 日训练，末尾 10 日复核 | 90.0848486531 | 5,760 |

T1/T2 的 A 拟合使用 2024-05-01 至 2025-10-22，评估 2025-10-23 至 2026-01-20。B 将 A 全并入拟合，评估 2026-01-21 至 2026-04-20。T3 早窗评估 2026-02-18 至 02-24、2026-03-05 至 03-11，分别用 20 天和 35 天目标场站历史。主窗口训练 2026-01-29 至 03-19，评估 03-20 至 03-29。实际日期从 `configs/warm_date_split_v1.csv` 和 `scripts/batch_energy_short.py::t3_windows` 读取。

## 配方、代码位置和关键服务器产物

### T1 v4

- 原发布提交 `924075540bafe2545d3fd2f975a54c2c7dd064d3`，入口 `scripts/run_phase_anchor.py`，主要函数 `historical_phase`、`phase_forecasts`、`t1_study`。
- T1 v3 原树加天气分支由 `scripts/run_t1_dynamic_gradient.py`、`scripts/t1_dynamic_wind.py` 和 `scripts/run_t1_issue_wind_error.py::dynamic_baseline` 支持。常规场站按 80% 动态梯度树、20% 三源订正风速功率分支融合。KNTBH4 为 64% 原全天树、16% 日 RMSE 全上下文网络、20% 同一风速分支。
- v4 对原预测加 25% 对称邻点平滑增量。邻点各占一半，站日内增量先中心化，边缘用同日边值。历史两个子块各至少 15 个计分站日、且选出一致的有益相移方向时，场站才进入平滑名单。实际名单从 `/workspace/qhl/runs/phase_anchor_cpu_v1/T1/phase_fit.json` 读取。KNTBH4 不参与平滑。
- 接受的 A 文件是 `/workspace/qhl/runs/phase_anchor_cpu_v1/T1/A/forecasts/symmetric_smoothing.csv`。B 是同一运行下的 `T1/B/<seed>/candidate.csv`，报告 `T1/B/b_score.json`。依赖的 v3 运行是 `/workspace/qhl/runs/t1_dynamic_gradient_gpu_v1`，平滑规则依赖 `/workspace/qhl/runs/strict_warm_replay_gpu_v1` 的历史折外预测。

### T2 v3

- 原发布提交 `24c30e1744f8127470374822b8f68991342a178c`，入口 `scripts/run_fusion_mechanisms.py::t2_predict`。用 `scripts/run_steamlink_followup.py` 的 `t2_components`、`station_calibration` 和 `energy_for` 获取各分量。
- v2 是 75% MLP 曲线积分与 25% 相对误差树，融合后逐站乘法校准。校准预测由拟合期更早部分生成，校准标签仅取拟合期尾部 20%。逐站系数为 `clip((sum(r)+20)/(sum(r*r)+20),0.6,1.2)`，其中 `r=预测电量/真实电量`。相关底模与校准回执在 `/workspace/qhl/runs/steamlink_followup_gpu_v1/T2`。
- v3 在逐站校准后使用 90% v2 和 10% **T1 v3** 的 96 点积分。T1 功率求和除以 4 转 MWh，保持实际电量范围。已采纳的锚点版本为 T1 v3。
- 接受的 A 文件是 `/workspace/qhl/runs/fusion_mechanisms_cpu_v1/T2/A/forecasts/t1_v3_integral_10.csv`。B 是同一运行下的 `T2/B/<seed>/candidate.csv`，报告 `T2/B/score.json`。T1 积分来自 `/workspace/qhl/runs/t1_dynamic_gradient_gpu_v1`。

### T3 v3

- 原发布提交 `24c30e1744f8127470374822b8f68991342a178c`，入口 `scripts/run_fusion_mechanisms.py::t3_predict`。交接版本还可直接阅读 `t3_components`、`t3_shape_blend` 和 `scripts/run_t3_nearby_donors.py::v3_reference`。
- 原始分量为六个短历史站的独立 XGBoost 短站树，以及用长站数据训练后冻结的时间网络和目标站岭头。当前岭强度为 0.20。岭头和重放入口在 `scripts/run_daily_bias_ridge.py`、`scripts/screen_short_transfer.py`，短站树在 `scripts/batch_energy_short.py`。
- 按站日令 `d=岭预测−短站树预测`，最终公式为 `clip(短站树 + 0.25 * mean(d) + 0.375 * (d−mean(d)), 0, 24.1026)`。现行权重为日均 25%、中心化的日内形状 37.5%。
- 接受的早窗文件为 `/workspace/qhl/runs/fusion_mechanisms_cpu_v1/T3/<inner_feb 或 inner_mar>/forecasts/shape_37_5.csv`。固定 50/10 为 `T3/B/<seed>/candidate.csv`，报告 `T3/B/b_score.json`。
- 分量依赖 `/workspace/qhl/runs/daily_bias_ridge_gpu_v1/T3` 的 0.20 岭，`t3_transfer_confirmation_gpu_v1/T3` 的固定窗分量，`same_issue_followup_v1/T3` 的早窗冻结网络，及 `three_track_gpu_batch_v1/T3` 的早窗短站树。各文件以所在目录的预测锁和回执为准。

## 数据、评分和从头运行

- 统一数据根 `/workspace/qhl/data`。日期目录为 `train/YYYYMMDD_0000`，本地标签文件名 `power_target.csv`。评估名单来自 `submission/T1_dataset/submission_template.csv` 等三个模板。
- 原底模环境为 `/workspace/ctt/model_augmentation/final_check/.venv/bin/python` 下的 XGBoost 3.0.5 CUDA。底模复训使用 `/workspace/cpy/code` 和 `/workspace/cpy/code/workdir` 的同伴实现及天气缓存，CPU 四线程，GPU 串行。需要神经底模重训时已有脚本会寻找单独的 PyTorch 环境，相关选择见 `scripts/compare_t2_mlp.py::select_mlp_runtime`。
- `scripts/score_forecasts.py` 同时输出 README 主分和 PDF 截零口径。标签有效条件是功率有限且 `power_limit=1`。T1/T3 每站日至少 72 个有效点，按站日 RMSE 和场站等权汇总；T2 实际电量为有效功率均值乘 24，仅计至少 72 个有效点且实际电量达到 24.1026 MWh 的站日。
- 若要重做原融合实验，可用原发布提交及服务器已有底模。原批次命令分别为 `scripts/run_phase_anchor.py --device cpu --threads 4 --out <新目录>` 和 `scripts/run_fusion_mechanisms.py --device cuda --threads 4 --out <另一个新目录>`，使用上述 Python。该融合入口的设备值用于核对原底模回执，组合本身只使用 CPU。若底模或回执丢失，按其原训练脚本和日期边界补建相应依赖。
- 上述分数均为已登记的本地验证。匿名官方批次的最终全量训练、模板导出及线上结果另行核对。本地评分 CSV 使用日期和本地键。
