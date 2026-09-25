# 第二项工作实验索引

本目录验证两个变量：纵向事件聚合，以及由长期 Federate 分区、Active-dependency、安全
frontier 和可撤销 lease 组成的横向优化。设计、因果边界和实现见
[`docs/WORK_2_DESIGN.md`](../docs/WORK_2_DESIGN.md)。

## 优化后消融

| 数据集 | Ranks | 横向实现 | 仅纵向 | 仅横向 | 组合 | 报告 |
|---|---:|---|---:|---:|---:|---|
| LULESH-64 | 64 | 8 Federates × 8 ranks，Active + frontier + lease | **2.399×** | 0.888× | 2.350× | [报告](atlahs_ablation/lulesh64_frontier_partitioned/REPORT.md) |
| HPCG-8 | 8 | 2 Federates × 4 ranks，Active + frontier + lease | 1.239× | 1.000× | **1.379×** | [报告](atlahs_ablation/hpcg8_frontier_partitioned/REPORT.md) |
| ICON-8 | 8 | 2 Federates × 4 ranks，Active + frontier + lease | 1.242× | 1.171× | **1.570×** | [报告](atlahs_ablation/icon8_frontier_partitioned/REPORT.md) |
| Grok-314B-256 | 256 | 8 Federates × 32 ranks，Active + frontier + lease | 2.381× | **1.500×** | **8.574×** | [报告](atlahs_ablation/grok256_frontier_partitioned/REPORT.md) |

四个数据集、每个单元三次重复，共 48 次正式运行。Grok 和 ICON 的横向单开获得端到端收益，
HPCG 持平，LULESH 的横向固定成本仍高于协调收益。完整汇总见
[最终四工作负载报告](atlahs_ablation/SUITE_REPORT.md)。

另外完成了三个 64-rank 大规模 HPC trace：HPCG-64（1,037 万操作）、ICON-64（310 万操作）和
LAMMPS-64（102 万操作）。它们的组合加速分别为 2.307×、2.228× 和 2.013×，报告见
[HPCG-64](atlahs_ablation/hpcg64_frontier_partitioned/REPORT.md) 和
[ICON-64](atlahs_ablation/icon64_frontier_partitioned/REPORT.md)、
[LAMMPS-64](atlahs_ablation/lammps64_frontier_partitioned/REPORT.md)。

## 优化前端到端消融

| 数据集 | 类型 | Ranks | 原始操作 | 仅纵向 | 仅横向 | 组合 | 报告 |
|---|---|---:|---:|---:|---:|---:|---|
| LULESH-64 | HPC | 64 | 11,438,062 | **2.421×** | 0.551× | 1.138× | [报告](atlahs_ablation/REPORT.md) |
| HPCG-8 | HPC | 8 | 626,216 | **1.280×** | 0.736× | 1.056× | [报告](atlahs_ablation/hpcg8/REPORT.md) |
| ICON-8 | HPC | 8 | 295,072 | **1.257×** | 0.768× | 1.065× | [报告](atlahs_ablation/icon8/REPORT.md) |
| Grok-314B-256 | 大模型训练 | 256 | 114,000,750 | **2.397×** | 0.720× | 1.577× | [报告](atlahs_ablation/grok256/REPORT.md) |

这组历史结果的横向实现为一 rank 一 JVM、短周期时间请求；关闭时只有一个集中式计算
Federate。它们用于说明固定进程成本为何曾经抵消协调收益。完整汇总见
[四工作负载报告](atlahs_ablation/SUITE_REPORT.md)。

## 协调层上界实验

[`active_dependency_scaling/grok_rank_local`](active_dependency_scaling/grok_rank_local/REPORT.md)
保留了排除 GPUSim、ns-3 和网络模型后的纯协调实验。256 Federates 下横向单开为 2.153×，
用于解释协调算法的潜力，不作为端到端消融结果。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /path/to/trace.goal --output /path/to/output --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py \
  --output /path/to/output --run-only --repeats 3 \
  --ranks-per-federate 32 --timeout 3600 --quiet-worker-logs
python3 experiments/atlahs_ablation/summarize_suite.py
```

中断后可用 `--repeat-start N` 复用先前的 `result.json`，从第 N 次重复继续。原始 GOAL traces、
逐次控制面日志和构建产物不进入 Git；仓库保存脚本、数据集元数据、最终汇总和报告。
