# 复现说明

## 1. 数据获取与许可

本仓库不重新分发第三方原始数据。复现实验使用的数据及锁定版本如下：

| 数据角色 | 数据标识 | 锁定版本 | 许可 |
|---|---|---|---|
| 通用可见光参考 | `coco2017_person` | COCO 2017 train 与 trainval annotations | COCO 标注 CC BY 4.0；图像许可逐项核验 |
| 煤矿可见光源 | `dslmfplus_coal_miner_v1` | Figshare file 40215118 | CC0 1.0 |
| 煤矿可见光目标 | `dsdpm66_coal_miner_v1` | Figshare file 47348143 | CC BY 4.0 |
| 红外/可见光输入 | `llvip_v1` | official Google Drive release | LLVIP non-commercial research terms |
| 甲烷时序 | `mendeley_methane_v1` | DOI `10.17632/yd7vw4c5mk.1` | CC BY 4.0 |

完整来源决策、归档 SHA256、字段适配合同和最小样本记录位于：

- `evidence/data/dataset_source_candidates.reviewed.json`
- `evidence/data/dataset_source_decision.json`
- `evidence/data/source_reviews/`
- `evidence/data/license_snapshots/`
- `data/locked/canonical_dataset_registry.json`

下载后应先校验归档哈希，再运行 `tools/data/` 中对应的适配与审查脚本。不可变文件清单和分组划分分别为 `data/locked/file_manifest.parquet` 与 `data/locked/split_manifest.parquet`。

## 2. 环境复现

推荐使用隔离环境。GPU 训练前应根据本机 CUDA 与驱动安装匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,vision]"
```

Windows PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
```

已核验的完整版本组合见 `env/`。环境安装完成后运行：

```bash
python -m mining1_exp.cli version
python -m mining1_exp.cli validate-config --config configs/protocol_lock.pretest.yaml
pytest -q
```

## 3. 数据与协议核验

正式运行前至少核验以下文件：

```text
configs/protocol_lock.pretest.yaml
configs/artifact_contract.template.yaml
plans/experiment_steps.csv
data/locked/canonical_dataset_registry.json
data/locked/file_manifest.parquet
data/locked/split_manifest.parquet
data/locked/ontology_lock.yaml
data/locked/methane_role_lock.json
```

原始组、模型选择组、校准组和测试组必须保持分离。公开仓库未提供测试真值、测试密封文件及逐样本测试预测，避免对留出评价造成二次污染。

## 4. 执行入口

本地模块和最小样本检查通过后，完整训练通过 `slurm/` 中的入口提交。主要入口包括：

- `slurm/train_s1_gru_array.sbatch`
- `slurm/train_t1_array.sbatch`
- `slurm/train_v2_pretrain_array.sbatch`
- `slurm/train_v2_finetune_array.sbatch`
- `slurm/fit_s1_baselines.sbatch`
- `slurm/measure_efficiency.sbatch`

所有入口均通过 `python -m mining1_exp.cli` 调用固定工作流。资源、路径和分区参数应根据目标 Slurm 集群调整，不应直接复用本项目的本地绝对路径。

## 5. 结果核验

最终结果文件及 SHA256：

| 文件 | SHA256 |
|---|---|
| `results/S1/metrics.parquet` | `f33c3e2d105b3a365fbacc9ab081cd01aedf92d33feb3dc61d8d3780ebfa4369` |
| `results/C1/efficiency.parquet` | `b3ee40d7d36e6a01848cbe82aa8762f8aee9d82257ef7b2a0f9a436b0f6bbabb` |
| `results/C1/efficiency_scope.csv` | `c07bcc650167a6932d5329a9ccc90cd387e6fa2220da62a3760187c6b7a3111f` |
| `results/final/aggregate.parquet` | `c44d2f22b1994e5cde23376d65b614237588e26b6e286697cc39c5cdba3a6557` |
| `results/final/statistical_tests.json` | `b2262092df4aee734639b9072eb6d443c07d239f0be1cd29a1be589cd969ea55` |
| `results/final/claim_status.csv` | `f75cffde4d90faa85587832f03443cb86c76ccadf649ecfc512b0755d7fb727e` |

`results/final/claim_status.csv` 是解释结果范围的首要入口。状态为 `removed` 或 `negative` 的分支应按文件中的允许表述解释，不应转换为正向性能结论。

## 6. 发布包边界

本仓库用于代码审阅、协议复核和聚合结果复现。论文正文、投稿文件、第三方原始数据、模型权重、测试真值及逐样本预测由作者单独管理，不属于本公开提交。
