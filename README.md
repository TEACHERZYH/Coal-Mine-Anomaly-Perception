# Coal-Mine Anomaly Perception

本仓库提供“公开异构证据驱动的煤矿异常感知表征预训练与离线迁移评测”研究的精简复现材料。公开内容包括实验代码、固定协议、数据适配与核验脚本、Slurm 入口、自动化测试、不可变数据清单以及论文结果对应的聚合表。

## 研究范围

当前公开证据支持以下内容：

- 煤矿可见光、非煤矿可见光/红外和甲烷时序公开数据的来源核验、适配、本体映射与分组划分。
- 可见光与热红外输入分支的有效性检查。
- 甲烷未来风险评分中 GRU、HGB 和固定规则参考的离线比较。
- S1-GRU 在固定 Slurm 3090 环境中的离线效率测量。

最终聚合结果保留了负面或条件不足的结论。固定协议下，GRU 的甲烷未来风险事件宏 F1 相对规则参考低 67.7 个百分点。多源视觉迁移、代理鲁棒性、事件融合、图消息传递和事件记忆未形成满足可比条件的定量结果。

## 仓库内容

- `mining1_exp/`：数据、训练、预测、评价、治理和最终聚合代码。
- `configs/`：协议、产物合同和实验矩阵配置。
- `plans/experiment_steps.csv`：最终实验步骤及状态。
- `tools/`：数据核验、环境检查和远程执行辅助脚本。
- `slurm/`：训练、预测、评价和效率测量的 Slurm 入口。
- `tests/`：模块、合同、数据隔离和流程测试。
- `env/`：本地与远程环境锁定清单。
- `data/`：经选择公开的不可变清单、本体和数据适配回执，不含原始数据或测试真值。
- `results/`：聚合指标、统计结果和效率结果。
- `evidence/`：数据来源、最终门状态、资源试验摘要和最小远程环境 smoke 回执的精简子集。

## 明确排除

根据仓库发布范围，本仓库不包含：

- 论文 Markdown、Word、PDF、投稿模板及其渲染产物；
- 第三方原始数据集或下载归档；
- 测试真值、测试密封文件和逐样本预测；
- 模型 checkpoint、校准器及训练缓存；
- 完整远程作业日志、失败快照、认证凭据和内部监控记录；仅保留测试套件所需的最小环境 smoke 回执。

原始数据须按各数据集许可从其官方来源获取。数据版本、许可、归档哈希和适配规则见 `REPRODUCIBILITY.md`、`evidence/data/` 与 `data/locked/canonical_dataset_registry.json`。

## 环境

代码支持 Python 3.9 及以上版本。已核验环境位于：

- `env/local-dl_env-core.lock.txt`
- `env/remote-mining1-py39-cu121.P020_v1_ort1201.lock.txt`

PyTorch/CUDA 应按目标硬件从 PyTorch 官方渠道安装，再安装本项目：

```bash
python -m pip install -e ".[test,vision]"
python -m mining1_exp.cli version
pytest -q
```

## 结果入口

- `results/S1/metrics.parquet`：甲烷时序模型与规则参考的逐重复聚合指标。
- `results/C1/efficiency.parquet`：S1-GRU 离线效率测量。
- `results/final/aggregate.parquet`：最终聚合表。
- `results/final/statistical_tests.json`：统计检验与区间结果。
- `results/final/claim_status.csv`：各研究结论的证据状态与允许表述。

## 引用

论文正式发表后，将在本节补充题名、作者、期刊、年份和 DOI。当前作者及投稿元数据未写入仓库。
