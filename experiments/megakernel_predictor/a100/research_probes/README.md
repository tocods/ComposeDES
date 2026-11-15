# A100 选题验证

## 结论

优先推进执行图感知的性能预测。固定算子、缓冲区与依赖时，CUDA Graph 节点创建顺序足以改变 H2D 配额的优劣。使用图内计时点时，若干排序翻转消失。稀疏 KV 方向保留为后备。在当前单层、固定选中 KV 的负载中，简单拆批的实际收益有限，保留原读回速率的理想重叠空间约为整层时间的 4%–6%。

实验设备为 NVIDIA A100 PCIe 40 GB，驱动 595.71.05，PCIe Gen4 ×16，CUDA 13.0.88，PyTorch 2.14.1，SGLang 0.5.23，FlashInfer 0.7.2。表格为 GPU 端时间。两组实验使用不同的算子组合与计时流程。

## 1. 节点创建顺序能否改变候选排序

### 实验设置

同一数据依赖图有两条串行链。读回链从 CPU 加载六组 KV，计算链依次执行七项计算。第一项计算读 GPU 驻留数据，此后每项计算依赖前一项计算和对应的读回。每组读回 4,096 个 BF16 KV token，六组合计 96 MiB。

- 第一套流水使用 SGLang HiCache 搬运和 FlashInfer FA2。使用 Qwen3‑14B 的 attention 头尺寸，追加 query 数量取 128、512、1,536。[Qwen3‑14B 官方配置](https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json)
- 第二套流水保留同样的搬运量，用 BF16 GEMM 消费输入。矩阵输出宽度为 1,024、3,072、8,192，并加入 CUDA 异步 DMA。
- 每个算子捕获一次。在同一进程内，复用相同 kernel 函数、参数和缓冲区地址，以三种合法的拓扑顺序重建 CUDA Graph。运行时按随机顺序交错测不同候选。每个候选每组 18 次，独立进程重复两组，共 5,832 条。正确性检查在必读 KV 区域填入 NaN 后回放，要求全输出逐元素一致。
- 每次回放以外部首尾 CUDA event 计时。插桩对照在每项逻辑任务前后额外插入 event 节点，保留任务间的功能依赖。单独使用 Nsight Systems 采集部分候选的 kernel 时间线，图中仍只有原有功能节点。

### 配对结果

表中均为同轮 `T(64 CTA) − T(8 CTA)` 的中位数。正数表示 64 CTA 更慢。合并两轮的区间使用按进程分层的配对 bootstrap。以两个进程的中位方向一致、合并区间不跨零、合并差值超过 50 µs 为描述性筛选，6 个形状有 4 个出现排序翻转。50 µs 为分析时采用的实效阈值，多个形状的区间未做多重比较校正。

| 流水 | 先创建搬运节点 | 交错创建计算与搬运节点 | 交错创建并增加图内事件 |
| --- | ---: | ---: | ---: |
| 128 query attention | +0.120 ms | −0.056 ms | +0.169 ms |
| 512 query attention | +0.649 ms | −0.169 ms | +0.727 ms |
| 1,536 query attention | +3.517 ms | +0.812 ms | +2.952 ms |
| GEMM，宽度 1,024 | +0.008 ms | −0.021 ms | +0.113 ms |
| GEMM，宽度 3,072 | +0.147 ms | −0.180 ms | +0.414 ms |
| GEMM，宽度 8,192 | +0.769 ms | −0.335 ms | +1.059 ms |

512-query attention 的前两格 95% 区间分别约为 [+0.540, +0.796] ms 和 [−0.252, −0.078] ms。8,192 宽度 GEMM 分别为 [+0.762, +0.817] ms 和 [−0.376, −0.284] ms。在这两种顺序间，各 GEMM 形状的 DMA 候选中位数变化低于 0.01 ms。GEMM 表里的翻转针对 8 与 64 CTA 这对候选，候选池中还包含 DMA。

### 目前能定位到的机制

独立的 Nsight Systems 采样中，512-query attention、64 CTA 先创建搬运节点时，首个 H2D kernel 比首个 attention kernel 早 0.982 ms 启动。交错创建时，首个 attention kernel 早 0.204 ms 启动。两种顺序的六个搬运 kernel 累计分别约为 8.231 和 8.232 ms。采样显示了并发时序确实变化。采样环境与正式配对测量的延迟单独记录，时间线无法进一步分解 SM、L2、HBM 各自的影响。

CUDA 已说明节点创建顺序会参与并发节点的调度。这里发现的是这一行为在两条实际搬运与计算流水上造成了可重复的候选排序翻转，以及把图内插桩结果迁移到生产图会选错的具体例子。样本目前来自一张卡、一版驱动与一套依赖结构。本实验未运行任何完整模型或比较论文系统的预测准确率。[NVIDIA 的 CUDA Graph 调度说明](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) · [One Megakernel is Not Enough](https://arxiv.org/abs/2610.35336)

### 对 ComposeDES 的含义

设相同的功能依赖图为 \(D\)，节点创建顺序为 \(\pi\)，在两个搬运配置间的实测差值为 \(\Delta(\pi)\)。对于只依赖 \(D\) 和隔离算子耗时、对所有顺序给同一个差值 \(\widehat\Delta\) 的预测器，至少一个顺序的误差满足

\[
\max_i |\widehat\Delta-\Delta(\pi_i)|
\geq \tfrac12 |\Delta(\pi_1)-\Delta(\pi_2)|.
\]

在 512-query attention 上，实测中位数给出约 0.409 ms 的误差下界。用两种顺序共同标定一个差值也无法同时给出正确排序。

下一步最小的算法验证是在 ComposeDES 里同时输入功能依赖和节点创建顺序，显式表示两条资源队列，在零或少量目标图首尾测量下校准释放时刻和并行干扰。应冻结标定负载，在不同依赖拓扑以及新的计算形状上比较选中配置相对事后最快配置的差距。查表、等预算直接搜索、忽略创建顺序的阶段模型使用相同数目的首尾测量；固定构图顺序也应作为基线。验证目标是用更少的目标图测量选对更多配置，当前数据尚未验证这项算法收益。

## 2. 稀疏 KV 若先处理已就绪请求，还能省多少

使用 SGLang 原生 block-selected MHA HiSparse Resolve 以及 FlashInfer FA2。整批基线等待同层所有 Resolve 完成后调用一次 attention，这对应所测的同步 Resolve 路径。每个请求选中 64 个 16-token 块，32 个 query 头、8 个 KV 头、头维 128。固定选中块和缓存内容，只改变缺页分散在各请求还是集中在少数请求，第三种分布留出一部分全命中请求、同时保留少数慢请求。总体缺页率均为 25%，搬运量在 batch 8/32 时为 8/32 MiB。比较半批、四分之一批和全命中请求优先。每次回放前恢复 KV 和缓存元数据。验证 Resolve 输出的映射、K/V 数据和 attention 输出。每个候选在每种构图顺序下有 48 次交错测量，共 4,032 条，含隔离与逐组计时。

本机安装版本的 HiSparse MHA Resolve 在每块 256 线程和该 1,024-token 选择量下触发了 shared memory 越界。所有测量路径一致使用每块 1,024 线程的同一原生 kernel，未修改 SGLang 源码。调整后，batch 8 的三种缺页分布均通过 Compute Sanitizer 的 Resolve 检查。基线覆盖原生 Resolve 和 attention 的单层等待结构，未包括真实索引计算、跨层预取、连续批处理或 SGLang 服务端。

| Batch | 缺页分布 | 整层基线时间 | 拆成相等两批的配对收益 | 已测最佳拆批的配对收益 | 理想重叠空间 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 均匀 | 0.759 ms | −11.3 µs | −11.3 µs | 5.5% |
| 8 | 集中 | 0.738 ms | −3.1 µs | 0.0 µs | 4.3% |
| 8 | 少数请求慢 | 0.736 ms | −5.1 µs | −5.1 µs | 4.3% |
| 32 | 均匀 | 2.866 ms | +41.0 µs | +41.0 µs | 4.1% |
| 32 | 集中 | 2.876 ms | +37.9 µs | +77.8 µs | 4.4% |
| 32 | 少数请求慢 | 2.875 ms | +41.0 µs | +41.0 µs | 4.4% |

拆批配对结果使用先创建搬运节点的图；该顺序下两组同功能图的整层基线只差约 1 µs。均匀分布的全命中优先方案退化为半批，因此表中只保留不同功能的候选。集中缺页形状中全命中优先的 +77.8 µs 在交错创建顺序下降为 +19.5 µs，后者弱于固定半批的 +31.7 µs。不能用事后挑中的 2.7% 当作冻结策略的收益。

理想估计保持整批 Resolve 的累计服务时间 \(C\) 和整批 attention 的计算时间 \(A\)。将 \(A\) 均摊到 \(B\) 个请求，假设其余 \(B-1\) 个请求从一开始即可计算，最后一个在 \(C\) 时刻就绪，则理想完成时刻为 \(\max(A,C+A/B)\)。估计允许完全重叠、零拆批开销及无争用，且保持原读回时间。它刻画该模型下重排计算的收益空间。改变搬运算法、减少字节数，或者让少数请求独立推进多个层，都超出这个估计的范围。结果 JSON 也包含根据逐组读回时刻进行的回放，供分析，但该回放使用图内计时的到达时刻。

当前数据支持继续寻找跨请求阻塞更明显的真实稀疏负载。要开展这条路线，优先比较真实到达轨迹下可提前完成的请求数、逐请求延迟和整批吞吐，与固定半批及已有重叠实现对照。HiSparse 的跨层预取、FlashOffloading 的 kernel 内重叠与 SolidAttention 的细粒度 I/O 调度均应纳入相关工作；本实验只实测了上述 SGLang kernel 基线。[HiSparse](https://arxiv.org/abs/2608.07009) · [FlashOffloading](https://arxiv.org/html/2611.07542v1#S5.SS4) · [SolidAttention](https://www.usenix.org/system/files/fast26-zheng.pdf)

## 复现

在此目录使用 CUDA 13 构建图回放库并运行。结果目录可按本地仓库实验布局设置。

```bash
nvcc -std=c++17 -shared -Xcompiler -fPIC explicit_graph.cu -lcuda -o libexplicit_graph.so
mkdir -p ../../runs/mpk-a100/research-probes
for session in A B; do
  python order_probe.py --session "$session" --repeat 18 --output "../../runs/mpk-a100/research-probes/order-$session.jsonl"
  python sparse_probe.py --session "$session" --repeat 24 --output "../../runs/mpk-a100/research-probes/sparse-$session.jsonl"
done
nsys profile --trace cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o ../../runs/mpk-a100/research-probes/order-node \
  python order_probe.py --profile --repeat 5 --output ../../runs/mpk-a100/research-probes/order-node.jsonl
nsys export --type sqlite -o ../../runs/mpk-a100/research-probes/order-node.sqlite \
  ../../runs/mpk-a100/research-probes/order-node.nsys-rep
python analyze_probes.py ../../runs/mpk-a100/research-probes order sparse profile
```
