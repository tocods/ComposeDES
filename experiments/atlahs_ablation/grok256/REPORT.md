# ATLAHS Grok-314B 256-GPU 纵向 × 横向消融实验

实验日期：2026-09-24（Asia/Shanghai）

## 结论

纵向优化采用 Critical-path event acceleration。在 256-rank Grok-314B 派生工作流中，
它将计算事件从
32,768 个精确收缩到 4,096 个。关闭组件 stdout 落盘后，三次运行的墙钟中位数从
101.872 s 降至 48.946 s，缩短 52.0%，即 2.081 倍加速。

横向优化采用 rank-visible Federate 切分与 Active-dependency 协调。完整三 Federate 模型
没有暴露逐-rank 时间线；优化 Active-dependency 的高频实现后，单独开启时墙钟中位数为 101.320 s，比同批全关闭
快 0.54%；双开为 49.193 s，比全关闭快 2.071 倍，比只开启事件加速慢 0.50%。这两个差值
已接近运行波动范围，因此当前三联邦成员模型中的 Active-dependency 可视为性能基本持平，
同时仍减少 0.65% 和 0.23% 的时间授权。

## 数据集

- 官方文件：`grok.goal`
- 本地路径：`/home/sdic/atlahs/data/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal`
- 文件大小：8,410,598,614 bytes（7.833 GiB）
- SHA-256：`dec250a489ef317757de0aa58c1d0becb7063c8b7d2927bf349d6b2d54830c06`
- 256 ranks，114,000,750 条 GOAL 操作
- 59,710,454 个 `calc`、27,145,148 个 `send`、27,145,148 个 `recv`

转换参数与 LULESH-64 实验一致：每 rank 16 个 phase，每 phase 8 个计算事件。每个 phase
构成一个同主机、线性、零通信的 exact closed region。

- 基线计算事件：32,768
- 加速后计算事件：4,096
- exact regions：4,096
- 聚合网络边：4,758
- 模型化发送量：9,487,384,518,016 bytes
- 原始 trace 发送量：10,481,131,527,552 bytes
- 256-host chain，56 Gbit/s，单链路延迟 1,000 ns
- ns-3 store-and-forward flow macro

最后一个 phase 的发送没有下一 phase 可连接，因此没有纳入模型化发送量。数据准备耗时
160.35 s，采用两遍流式扫描，未将 1.14 亿条操作整体装入内存。

## 三次重复结果

组件 stdout 使用 `--quiet-worker-logs` 丢弃，控制面事件、协调指标和优化报告仍完整保留。
墙钟列为中位数，括号内为均值 ± 样本标准差。

| 纵向事件聚合 | 横向 Active 协调 | 墙钟 / s | 相对基线 | 计算事件 | 调度轮次 | FNCS grants | 控制面记录 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 101.872 (102.358 ± 1.209) | 1.000× | 32,768 | 1,117,671 | 1,119,194 | 73,554 |
| 关 | 开 | 101.320 (101.804 ± 0.839) | 1.005× | 32,768 | 1,111,667 | 1,111,885 | 74,579 |
| 开 | 关 | 48.946 (49.263 ± 0.594) | **2.081×** | 4,096 | 1,069,305 | 1,069,493 | 16,274 |
| 开 | 开 | 49.193 (48.994 ± 0.435) | 2.071× | 4,096 | 1,066,834 | 1,067,048 | 17,297 |

事件加速使计算事件减少 87.5%、控制面记录减少 77.9%，但调度轮次只减少 4.33%。这说明
Grok 的百万级轮次主要来自长逻辑时间和网络协调；墙钟收益主要来自减少计算后端事件、
序列化和控制面处理。

主动依赖单独使调度轮次减少 0.54%、grants 减少 0.65%，同时新增 1,019 次依赖更新。
实现优化后这些同步收益不再被调度器固定开销吞噬，但规模仍不足以形成显著墙钟加速。
这个机制的进一步收益取决于联邦成员数量、依赖图稀疏度以及保守授权占比。

## Active-dependency 实现优化

- Orchestrator 使用三个布尔量缓存依赖状态，只在 worker 事件改变 inflight 状态时构造和发布依赖图；空时间授权不再反复创建 Python `dict`、`set` 和排序签名。
- Broker 收到新 epoch 时将模拟器名称编译为整数邻接表并计算传递闭包；约百万个调度轮次只执行整数扫描。
- 调度器的 actionable time、lower bound 和 grant 缓冲区在轮次间复用，消除了每轮字符串、map/set 和 vector 重建。

优化前 Active 单开和双开的中位数分别为 110.625 s、55.748 s；优化后分别缩短至
101.320 s、49.193 s，对 Active 路径本身提升 1.092 倍和 1.133 倍。优化前结果保存在
`summary_pre_active_dependency_optimizer.json`，第一阶段优化结果也保存在
`summary_indexed_scheduler.json`。

曾验证让 orchestrator 在 Active 模式直接请求最大时间，以消除 1 ms 有界轮询。Grok 工作流
运行中出现空闲 GPUSim 被提前授权到终点、随后又收到新任务的情况，因此该方案未纳入最终实现；
当前继续保留有限 lookahead 作为动态派发窗口的因果安全边界。

## 正确性检查

- 四组均完成 4,758 个网络事件。
- 将宏任务名称归一化为 rank/phase 边界后，网络传输语义 SHA-256 完全一致。
- 模拟 makespan 位于 1,006,360,860,972–1,006,360,868,792 ns。
- 最大跨度 7,820 ns，即 0.00777 ppm，满足 1 ppm 等价阈值。
- 精确完成时间哈希不同，来自不同时间授权顺序的纳秒级偏移。

## 优化前默认日志首轮

保留 worker stdout 时，每个组合产生约 1.3–1.4 GiB 日志。首轮结果为 103.72、51.84、
115.98 和 59.91 s。它们来自 Active-dependency 高频路径优化之前，仅用于说明大量日志不是
原性能退化的主要来源。

默认日志结果位于 `summary_logged_smoke.json` 与 `runs_logged_smoke/`。

## 与 LULESH-64 对比

| 指标 | LULESH-64 | Grok-256 |
|---|---:|---:|
| 原始文件 | 665 MB | 8.41 GB |
| 原始 GOAL 操作 | 11,438,062 | 114,000,750 |
| 基线计算事件 | 8,192 | 32,768 |
| 基线墙钟中位数 | 13.597 s | 101.872 s |
| 事件加速比 | 2.234× | 2.081× |
| Active-dependency 单独效果 | 无稳定收益 | 优化后约持平 |

Grok-256 将原始操作规模扩大约 10 倍、仿真计算事件扩大 4 倍，并强化了事件聚合有效、
主动依赖在三成员 federation 中同步节省有限的结论。实现优化已经消除其明显退化。

## 二因素主消融

主消融固定每个 rank 对应一个 FNCS Federate，只改变两个因素：纵向事件聚合，以及横向
Active-dependency 协调。256-rank 四组合结果如下：

| 纵向优化 | 横向优化 | 墙钟 / s | 调度轮次 | 相对双关 |
|---|---|---:|---:|---:|
| 关 | 关 | 1.8262 | 32,654 | 1.000× |
| 关 | 开 | 0.8483 | 129 | 2.153× |
| 开 | 关 | 0.8609 | 4,090 | 2.121× |
| 开 | 开 | 0.6984 | 17 | **2.615×** |

Federate 切分负责暴露横向并行性，Active-dependency 负责利用它。为了保持四组的进程结构
和工作负载一致，切分方式在主消融中固定，不再作为第三个变量。原来的三 Federate 完整仿真
保留为架构旁证：其中逐-rank 时间线隐藏在 GPUSim 后面，因此横向优化单开只有 1.005×。

二因素定义、独立贡献和 8–256 ranks 扩展结果见
`experiments/active_dependency_scaling/grok_rank_local/REPORT.md`，主四组合机器可读结果见
`experiments/active_dependency_scaling/grok_rank_local/ablation_2x2.csv`。两个执行模型的补充数据
保存在 `experiments/atlahs_ablation/grok256/execution_model_comparison.csv`，不作为消融变量。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /home/sdic/atlahs/data/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal \
  --output experiments/atlahs_ablation/grok256 \
  --phases 16 --events-per-phase 8 --prepare-only

python3 experiments/atlahs_ablation/run_ablation.py \
  --output experiments/atlahs_ablation/grok256 \
  --run-only --repeats 3 --timeout 1800 --quiet-worker-logs
```

## 适用范围

实验使用全部原始操作累计计算时间和通信量，但按 phase/peer 聚合通信，并非 1.14 亿操作的
一对一重放。链式网络和 flow macro 也不是 Grok 训练集群的真实 Dragonfly 拓扑。当前结果
适合评价 ComposeDES 控制面机制的规模趋势；若评价绝对网络性能，需要接入原始拓扑并使用
更细粒度网络模型。

官方来源：

- https://spcl.inf.ethz.ch/Research/Scalable_Networking/ATLAHS/
- http://storage2.spcl.ethz.ch/traces/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/
