# 第二项工作实验索引

本目录验证两项性能机制：纵向事件聚合和横向 Active-dependency 时间协调。设计定义与实现
边界见 [`docs/WORK_2_DESIGN.md`](../docs/WORK_2_DESIGN.md)。

## 主实验

| 数据集/模型 | 用途 | 主要结论 | 报告 |
|---|---|---|---|
| ATLAHS LULESH-64 | 三 Federate 完整协同仿真 | 纵向 1.724×；横向约持平 | [`atlahs_ablation/REPORT.md`](atlahs_ablation/REPORT.md) |
| ATLAHS Grok-256 | 三 Federate 完整协同仿真 | 纵向 2.081×；横向 1.005× | [`atlahs_ablation/grok256/REPORT.md`](atlahs_ablation/grok256/REPORT.md) |
| Grok-256 rank-visible | 纵向 × 横向主消融 | 纵向 2.121×；横向 2.153×；双开 2.615× | [`active_dependency_scaling/grok_rank_local/REPORT.md`](active_dependency_scaling/grok_rank_local/REPORT.md) |

主消融固定每 rank 一个 FNCS Federate。纵向开关控制 exact phase event aggregation；横向
开关控制 Active-dependency 是否利用已经暴露的独立时间线。四组各重复三次并使用墙钟时间
中位数。

## Grok-256 主消融结果

| 纵向优化 | 横向优化 | 计算事件 | 调度轮次 | 墙钟 / s | 相对双关 |
|---|---|---:|---:|---:|---:|
| 关 | 关 | 32,768 | 32,654 | 1.8262 | 1.000× |
| 关 | 开 | 32,768 | 129 | 0.8483 | 2.153× |
| 开 | 关 | 4,096 | 4,090 | 0.8609 | 2.121× |
| 开 | 开 | 4,096 | 17 | 0.6984 | **2.615×** |

机器可读四组合位于
[`active_dependency_scaling/grok_rank_local/ablation_2x2.csv`](active_dependency_scaling/grok_rank_local/ablation_2x2.csv)，
8–256 ranks 扩展数据位于
[`active_dependency_scaling/grok_rank_local/results.csv`](active_dependency_scaling/grok_rank_local/results.csv)。

## 复现

完整协同仿真：

```bash
python3 experiments/atlahs_ablation/run_ablation.py --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py \
  --run-only --repeats 3 --timeout 900 --quiet-worker-logs
```

rank-visible 二因素实验：

```bash
python3 experiments/active_dependency_scaling/run_scaling.py \
  --ranks 8,32,64,128,256 --events-per-rank 128 --repeats 3
```

原始 GOAL traces、生成的输入、逐次运行日志、本地构建产物不进入 Git。仓库只保存复现脚本、
数据集元数据、最终汇总和报告。
