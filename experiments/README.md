# 第二项工作实验索引

本目录验证两个变量：纵向事件聚合，以及由完整 Federate 切分和 Active-dependency 组成的
横向优化。设计、因果边界和实现见 [`docs/WORK_2_DESIGN.md`](../docs/WORK_2_DESIGN.md)。

## 端到端消融

| 数据集 | 类型 | Ranks | 原始操作 | 仅纵向 | 仅横向 | 组合 | 报告 |
|---|---|---:|---:|---:|---:|---:|---|
| LULESH-64 | HPC | 64 | 11,438,062 | **2.421×** | 0.551× | 1.138× | [报告](atlahs_ablation/REPORT.md) |
| HPCG-8 | HPC | 8 | 626,216 | **1.280×** | 0.736× | 1.056× | [报告](atlahs_ablation/hpcg8/REPORT.md) |
| ICON-8 | HPC | 8 | 295,072 | **1.257×** | 0.768× | 1.065× | [报告](atlahs_ablation/icon8/REPORT.md) |
| Grok-314B-256 | 大模型训练 | 256 | 114,000,750 | **2.397×** | 0.720× | 1.577× | [报告](atlahs_ablation/grok256/REPORT.md) |

横向开启时，计算 Federate 数等于 ranks 且 Active-dependency 开启；关闭时只有一个集中式
计算 Federate。完整汇总见[四工作负载报告](atlahs_ablation/SUITE_REPORT.md)。

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
  --timeout 3600 --quiet-worker-logs
python3 experiments/atlahs_ablation/summarize_suite.py
```

中断后可用 `--repeat-start N` 复用先前的 `result.json`，从第 N 次重复继续。原始 GOAL traces、
逐次控制面日志和构建产物不进入 Git；仓库保存脚本、数据集元数据、最终汇总和报告。
