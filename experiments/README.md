# 第二项工作实验索引

本目录验证纵向事件聚合和横向 Active-dependency 时间协调。设计定义与实现边界见
[`docs/WORK_2_DESIGN.md`](../docs/WORK_2_DESIGN.md)。

## 跨工作负载补充实验

当前版本统一运行三个传统超算应用和一个大模型训练负载。四者均使用 orchestrator、GPUSim
和 ns-3 三 Federate 完整协同仿真，每个组合重复三次。

| 数据集 | 类型 | Ranks | 原始操作 | 仅纵向 | 仅横向 | 双开 | 报告 |
|---|---|---:|---:|---:|---:|---:|---|
| LULESH-64 | HPC 冲击流体 | 64 | 11,438,062 | **2.234×** | 0.996× | 2.234× | [报告](atlahs_ablation/REPORT.md) |
| HPCG-8 | HPC 多重网格/稀疏计算 | 8 | 626,216 | **1.305×** | 1.000× | 1.304× | [报告](atlahs_ablation/hpcg8/REPORT.md) |
| ICON-8 | HPC 天气与气候 | 8 | 295,072 | **1.290×** | 1.023× | 1.289× | [报告](atlahs_ablation/icon8/REPORT.md) |
| Grok-314B-256 | 大模型训练 | 256 | 114,000,750 | **2.081×** | 1.005× | 2.071× | [报告](atlahs_ablation/grok256/REPORT.md) |

完整汇总见[四工作负载报告](atlahs_ablation/SUITE_REPORT.md)，机器可读数据见
[`workload_summary.csv`](atlahs_ablation/workload_summary.csv) 和
[`all_workloads.csv`](atlahs_ablation/all_workloads.csv)。

四个完整模型都表明纵向聚合有效，加速范围为 1.290–2.234×。横向优化在当前三 Federate
架构中没有稳定收益，因为各 rank 的时间线隐藏在中央 GPUSim 后面。

## 横向扩展性实验

Grok-256 rank-visible 实验固定每 rank 一个 FNCS Federate，用于测量横向协调的性能上界：

| 纵向优化 | 横向优化 | 计算事件 | 调度轮次 | 墙钟 / s | 相对双关 |
|---|---|---:|---:|---:|---:|
| 关 | 关 | 32,768 | 32,654 | 1.8262 | 1.000× |
| 关 | 开 | 32,768 | 129 | 0.8483 | 2.153× |
| 开 | 关 | 4,096 | 4,090 | 0.8609 | 2.121× |
| 开 | 开 | 4,096 | 17 | 0.6984 | **2.615×** |

详细结果见 [Grok rank-visible 报告](active_dependency_scaling/grok_rank_local/REPORT.md)。该实验
排除了跨-rank 网络、GPUSim 和 ns-3，不能与完整协同仿真的绝对墙钟时间直接比较。

## 复现

每个完整模型先准备输入，再运行 2×2 实验：

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /path/to/trace.goal --output /path/to/output --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py \
  --output /path/to/output --run-only --repeats 3 \
  --timeout 900 --quiet-worker-logs
python3 experiments/atlahs_ablation/summarize_suite.py
```

rank-visible 扩展实验：

```bash
python3 experiments/active_dependency_scaling/run_scaling.py \
  --ranks 8,32,64,128,256 --events-per-rank 128 --repeats 3
```

原始 GOAL traces、生成输入、逐次运行日志和本机构建产物不进入 Git。仓库保存复现脚本、
数据集元数据、最终汇总和报告。
