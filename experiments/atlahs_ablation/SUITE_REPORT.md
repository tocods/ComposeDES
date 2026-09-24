# 三个超算应用与一个大模型训练负载的消融实验

实验日期：2026-09-24（Asia/Shanghai）

代码版本：FNCS `700d7d55a`、GPUSim `09861db84`、ns-3 `2f049f0d4`

## 工作负载

| 工作负载 | 类型 | Ranks | 原始 GOAL 操作 | 网络事件 |
|---|---|---:|---:|---:|
| LULESH-64 | HPC 冲击流体 | 64 | 11,438,062 | 17,032 |
| HPCG-8 | HPC 多重网格/稀疏计算 | 8 | 626,216 | 840 |
| ICON-8 | HPC 天气与气候 | 8 | 295,072 | 707 |
| Grok-314B-256 | 大模型训练 | 256 | 114,000,750 | 4,758 |

四个数据集均采用相同的三 Federate 完整协同仿真：workflow orchestrator、GPUSim 和 ns-3。
每 rank 划分 16 个 phase，每 phase 8 个计算事件；纵向开启后每个 phase 精确收缩为一个
宏事件。网络采用 56 Gbit/s、1,000 ns 链路的 store-and-forward flow 模型。每个 2×2
组合运行三次。

## 汇总结果

| 工作负载 | 双关 / s | 仅纵向 | 仅横向 | 双开 |
|---|---:|---:|---:|---:|
| LULESH-64 | 13.597 | **2.234×** | 0.996× | **2.234×** |
| HPCG-8 | 1.931 | **1.305×** | 1.000× | 1.304× |
| ICON-8 | 2.231 | **1.290×** | 1.023× | 1.289× |
| Grok-314B-256 | 101.872 | **2.081×** | 1.005× | 2.071× |

纵向事件聚合在四个工作负载上都产生明确收益，加速范围为 1.290–2.234×。横向
Active-dependency 在当前三 Federate 完整模型中均无稳定收益；三个 HPC 应用和 Grok 的
rank 内时间线都隐藏在中央 GPUSim 后面，broker 可利用的并行依赖较少。

Grok rank-visible 协调实验是横向机制的扩展性补充。在固定 256 Federates 时，横向单开为
2.153×，纵向单开为 2.121×，双开为 2.615×。该实验排除了跨-rank 网络与执行后端，不能
与本表的完整协同仿真墙钟时间直接比较。

## 正确性

四个工作负载的所有消融组合均保持相同的归一化网络语义哈希，且模拟 makespan 跨组合差异
均低于 1 ppm：LULESH 0.372 ppm、HPCG 0.0149 ppm、ICON 0.0505 ppm、Grok 0.00777 ppm。

机器可读汇总位于 `workload_summary.csv`，全部 16 个单元位于 `all_workloads.csv`。运行
`python3 experiments/atlahs_ablation/summarize_suite.py` 可从各数据集的 `summary.json`
重新生成这两个文件。
