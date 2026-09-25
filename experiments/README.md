# 第二项工作实验索引

本目录验证两个变量：纵向事件聚合，以及由长期 Federate 分区、Active-dependency、安全
frontier 和可撤销 lease 组成的横向优化。设计、因果边界和实现见
[`docs/WORK_2_DESIGN.md`](../docs/WORK_2_DESIGN.md)。

## 优化后消融

本轮正式实验同时记录总墙钟、启动段和稳态墙钟。启动段定义为进程启动到编排器完成 FNCS
初始化和初始 dispatch；稳态加速比只使用 marker 之后的墙钟，因此不把启动开销计入稳态性能。
实验启用了 1 ms 安全 frontier、broker 单订阅零拷贝、frontier-only epoch、worker 内部 quiet
logging，以及原始 JSON 事件日志复用。实验性异步 grant 默认关闭；完整消融使用已验证的保守
Active 协议。横向采用粗粒度长期 Federate：8 ranks 和 64 ranks 均切为 2 个 Federate，256
ranks 切为 8 个 Federate。

| 数据集 | Ranks | 横向实现 | 仅纵向稳态 | 仅横向稳态 | 组合稳态 | 报告 |
|---|---:|---|---:|---:|---:|---|
| LULESH-64 | 64 | 2 × 32 ranks，Active + frontier + lease | 3.166× | 1.004× | 3.165× | [报告](atlahs_ablation/lulesh64_frontier_partitioned/REPORT.md) |
| HPCG-8 | 8 | 2 × 4 ranks，Active + frontier + lease | 1.362× | 1.037× | 1.521× | [报告](atlahs_ablation/hpcg8_frontier_partitioned/REPORT.md) |
| ICON-8 | 8 | 2 × 4 ranks，Active + frontier + lease | 1.330× | **1.277×** | 1.869× | [报告](atlahs_ablation/icon8_frontier_partitioned/REPORT.md) |
| HPCG-64 | 64 | 2 × 32 ranks，Active + frontier + lease | 3.151× | 1.003× | 3.197× | [报告](atlahs_ablation/hpcg64_frontier_partitioned/REPORT.md) |
| ICON-64 | 64 | 2 × 32 ranks，Active + frontier + lease | 3.024× | 1.016× | 3.112× | [报告](atlahs_ablation/icon64_frontier_partitioned/REPORT.md) |
| LAMMPS-64 | 64 | 2 × 32 ranks，Active + frontier + lease | 2.932× | 1.013× | 3.030× | [报告](atlahs_ablation/lammps64_frontier_partitioned/REPORT.md) |
| Grok-314B-256 | 256 | 8 × 32 ranks，Active + frontier + lease | 2.760× | **1.464×** | 12.066× | [报告](atlahs_ablation/grok256_frontier_partitioned/REPORT.md) |

七个数据集、每个单元三次重复，共 84 次正式运行。网络完成语义在所有单元中一致，模拟
makespan 跨度均低于 10 ppm。横向在全部数据集上保持非负中位收益，但只有 ICON-8 和 Grok-256
达到明显加速：两者的调度轮次分别下降 80.12% 和 89.66%，且每条新增控制记录能消除 18.49 和
292.40 个轮次。其余数据集的同步窗口较短或跨分区依赖较密，减少的轮次不足以明显超过额外 JVM
与消息协调成本。定量解释、全部三次耗时和正确性结果见
[完整汇总](atlahs_ablation/SUITE_REPORT.md)，机器可读总表见
[workload_summary.csv](atlahs_ablation/workload_summary.csv)。

## 优化前端到端消融

| 数据集 | 类型 | Ranks | 原始操作 | 仅纵向 | 仅横向 | 组合 | 报告 |
|---|---|---:|---:|---:|---:|---:|---|
| LULESH-64 | HPC | 64 | 11,438,062 | **2.421×** | 0.551× | 1.138× | [报告](atlahs_ablation/REPORT.md) |
| HPCG-8 | HPC | 8 | 626,216 | **1.280×** | 0.736× | 1.056× | [报告](atlahs_ablation/hpcg8/REPORT.md) |
| ICON-8 | HPC | 8 | 295,072 | **1.257×** | 0.768× | 1.065× | [报告](atlahs_ablation/icon8/REPORT.md) |
| Grok-314B-256 | 大模型训练 | 256 | 114,000,750 | **2.397×** | 0.720× | 1.577× | [报告](atlahs_ablation/grok256/REPORT.md) |

这组历史结果的横向实现为一 rank 一 JVM、短周期时间请求；关闭时只有一个集中式计算
Federate。它们用于说明固定进程成本为何曾经抵消协调收益，保留在各数据集原目录中作为历史
对照。

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
  --ranks-per-federate 32 --frontier-update-quantum-ns 1000000 \
  --timeout 3600 --quiet-worker-logs
python3 experiments/atlahs_ablation/summarize_suite.py
```

8-rank 数据集把 `--ranks-per-federate` 设为 4；64-rank 和 256-rank 数据集设为 32。
复现正式消融时不要设置 `FNCS_ASYNCHRONOUS_GRANTS`，使实验性异步 grant 保持关闭。
中断后可用 `--repeat-start N` 复用先前的 `result.json`，从第 N 次重复继续。原始 GOAL traces、
逐次控制面日志和构建产物不进入 Git；仓库保存脚本、数据集元数据、最终汇总和报告。
