# 协同仿真 DAG 控制面重构方案

## 1. 决策摘要

采用“通用 FNCS broker + 中间件层 workflow-orchestrator + 领域仿真器 worker”的结构。

- `fncs_broker` 继续只负责发布订阅、消息转发和保守时间同步，不在 broker 内核中硬编码 AI 工作流语义。
- 新增 `fncs_workflow_orchestrator`，作为中间件层的一个 FNCS 联邦成员，持有唯一的 DAG 状态和全局工作流控制面。
- GPUSim 不再解析完整 DAG、递减依赖计数或发起网络任务，只执行中间件下发的计算任务并报告结果。
- ns-3 不感知 DAG，只执行中间件下发的网络传输并报告结果。
- 三个联邦成员都通过 FNCS broker 请求和获得逻辑时间，任何 worker 都不能自行推进全局工作流状态。

不直接把 DAG 调度写进 `broker.cpp`。broker 是通用时间协调内核，将工作流语义写入其中会导致协议、调度策略和 FNCS 一致性算法耦合，也难以单元测试和替换。

## 2. 控制权边界

| 能力 | 当前所有者 | 目标所有者 |
|---|---|---|
| DAG 解析与合法性检查 | GPUSim | workflow-orchestrator |
| 依赖计数、就绪队列、任务状态机 | GPUSim | workflow-orchestrator |
| 计算任务释放时机 | GPUSim | workflow-orchestrator |
| 任务到计算资源的全局放置策略 | GPUSim | workflow-orchestrator |
| 计算资源内部排队和执行时间演化 | GPUSim | GPUSim |
| 跨主机边生成和网络任务释放 | GPUSim | workflow-orchestrator |
| 路由、排队、拥塞、传输完成 | ns-3 | ns-3 |
| 重试、取消及工作流完成判定 | GPUSim | workflow-orchestrator |
| 计算故障的物理过程和资源状态 | GPUSim | GPUSim，并上报中间件 |
| 全局逻辑时间授权 | FNCS broker | FNCS broker |

全局放置与资源内部调度必须区分。中间件决定任务送到哪台 host、何时释放；GPUSim 仍负责模拟该 host 上的排队、共享、故障和实际完成时间。

## 3. 组件结构

```text
                         workflow.json / task catalog
                                      |
                                      v
                           +-----------------------+
                           | workflow-orchestrator |
                           | DAG + ready queue      |
                           | placement + lifecycle |
                           +-----------+-----------+
                                       |
                     FNCS JSON events + logical time
                                       |
                         +-------------+-------------+
                         |        fncs_broker        |
                         | pub/sub + time grant only |
                         +-------------+-------------+
                                       |
                    +------------------+------------------+
                    |                                     |
              +-----+------+                        +-----+-----+
              | GPUSim     |                        | ns-3      |
              | compute    |                        | network   |
              | worker     |                        | worker    |
              +------------+                        +-----------+
```

`workflow-orchestrator` 建议放在 FNCS 仓库中，生成独立可执行文件，而不是创建第四套通信库。它使用公开 `fncs.hpp` API，以普通联邦成员身份参与时间同步。

## 4. 工作流状态模型

### 4.1 计算任务状态

```text
BLOCKED -> READY -> DISPATCHED -> RUNNING -> SUCCEEDED
                                      |          |
                                      v          v
                                    FAILED     CANCELED
```

- `BLOCKED`：至少一个前驱或输入传输未完成。
- `READY`：所有输入条件满足，等待中间件放置和下发。
- `DISPATCHED`：已发送给 GPUSim，等待接收确认或完成事件。
- `RUNNING`：可选状态，由 GPUSim 的 started 事件确认。
- 终态只能由带匹配 `correlation_id` 的 worker 结果触发。

### 4.2 DAG 边状态

```text
WAIT_PRODUCER -> LOCAL_READY
              -> NET_DISPATCHED -> NET_COMPLETED
```

- 同 host 的边在生产者完成后直接进入 `LOCAL_READY`，不生成网络事件。
- 跨 host 的边由中间件生成唯一 `transfer_id` 并下发 ns-3。
- 一个子任务仅在全部输入边达到 `LOCAL_READY` 或 `NET_COMPLETED` 后进入 `READY`。

### 4.3 幂等要求

- 每个命令有全局唯一 `event_id`，每次执行尝试有唯一 `attempt_id`。
- worker 缓存已完成的 `event_id`；重复命令不得重复执行，只能重发原结果。
- orchestrator 对重复、过期、未知 attempt 的结果记录告警但不重复推进 DAG。
- 所有状态迁移追加写入事件日志，支持确定性回放和故障定位。

## 5. 事件协议

协议使用 JSON，不再使用当前的 `src?task?entity?dst?bytes` 和 `/` 拼接。每次 publish 的值是一个批次；同一逻辑时间、同一 topic 的多条事件合并为一个 JSON 数组。

### 5.1 通用信封

```json
{
  "schema_version": "2.0",
  "run_id": "run-20260814-001",
  "batch_id": "batch-000042",
  "logical_time_ns": 613756000000,
  "events": [
    {
      "event_id": "evt-000123",
      "kind": "compute.dispatch",
      "correlation_id": "task-r3-phase0-attempt1",
      "payload": {}
    }
  ]
}
```

### 5.2 固定 topic

| FNCS key | 发布者 | 订阅者 | 内容 |
|---|---|---|---|
| `orchestrator/compute/dispatch` | orchestrator | GPUSim | 计算任务批次 |
| `gpusim/compute/started` | GPUSim | orchestrator | 可选的开始确认 |
| `gpusim/compute/completed` | GPUSim | orchestrator | 成功、失败、取消及时间 |
| `orchestrator/network/dispatch` | orchestrator | ns-3 | 网络传输批次 |
| `ns3/network/completed` | ns-3 | orchestrator | 单流/集合通信结果 |
| `worker/status` | workers | orchestrator | ready、error、drained |
| `orchestrator/control` | orchestrator | workers | start、cancel、end |

不为每个任务动态创建 topic，避免订阅表和配置规模随 trace 线性增长。

### 5.3 计算命令最小字段

```json
{
  "task_id": "r3_phase0",
  "attempt": 1,
  "target_host": "host4",
  "release_time_ns": 613756000000,
  "deadline_ns": null,
  "compute": {
    "cpu": {"pes": 1, "length": 1128490000, "ram_mb": 0},
    "gpu": {"kernels": []}
  }
}
```

GPUSim 启动时只加载 hosts/resource 配置；task spec 随 dispatch 下发。过大的不可变 kernel 描述可后续改为 `task_spec_id + catalog_digest`，但首版应优先保证控制权真正迁移。

### 5.4 网络命令最小字段

```json
{
  "transfer_id": "edge-r3_phase0-r2_phase1",
  "collective_id": null,
  "src_task_id": "r3_phase0",
  "dst_task_id": "r2_phase1",
  "src_host": "host4",
  "dst_host": "host3",
  "bytes": 10240,
  "flow_id": "flow-00091"
}
```

集合通信不能在转换阶段压成一个普通点到点包。后续扩展字段为 `collective_id`、`collective_type`、`participants`、`algorithm` 和分片信息，由 orchestrator 展开为 flow DAG，或由 ns-3 的 collective adapter 展开；两者只能有一个权威展开者。

## 6. 时间同步规则

1. 所有 publish 必须发生在本联邦成员已获得的逻辑时间上。
2. 联邦成员先批量 publish 当前时间产生的结果，再调用 `fncs::time_request(next_local_event)`。
3. orchestrator 没有内部定时事件时请求无穷大，由 worker 的更早完成事件将其唤醒。
4. orchestrator 获得时间 `T` 后，先消费所有输入批次，再一次性完成状态迁移，最后批量发布 `T` 时刻的新命令。
5. worker 收到 dispatch 后，只在本地事件队列中安排执行；完成前不得代替中间件解锁 DAG 后继。
6. 对同一时间的连锁事件采用 FNCS barrier 轮次，不通过篡改时间戳绕过因果关系。若 FNCS 2.3.2 无法稳定处理零时间反馈，则协议明确使用 `time_delta=1ns` 形成最小因果间隔。
7. 终止由 orchestrator 判断。只有 DAG 全部终态、无在途 compute/network attempt 时，才发布 `orchestrator/control=end`。

批处理实现为 `Map<(logical_time, topic), List<Event>>`。每次进入 `time_request` 前，把每组编码为一个 JSON batch 并 publish 一次；接收方解析 batch 后逐事件处理。这保留每条事件的身份，不会把多个完成错误地视为同一个完成。

## 7. 迁移步骤

### Phase 0：冻结可运行基线

- 为 FNCS、GPUSim、ns-3 建立 `cosim/dag-control-plane-v2` 分支。
- 提交当前协同通信源码，不提交 trace、实验输出和本机构建产物。
- 记录每个仓库 commit、工具链和已知基线结果。

### Phase 1：协议库和 orchestrator 骨架

- 在 FNCS 新增 `src/workflow/` 和 `fncs_workflow_orchestrator` target。
- 实现 JSON schema、DAG loader、拓扑排序、状态机、批量 publisher 和事件日志。
- 用 fake compute/network worker 做纯中间件单元测试，不依赖 GPUSim/ns-3。

### Phase 2：GPUSim worker 模式

- 新增 `--mode worker`，启动时只读取 hosts 和 fault/resource 配置。
- 新增动态 `ComputeTaskFactory`，把 dispatch payload 转为单个 `GpuJob`。
- 计算返回只发布 completed；移除 worker 模式下的 `remainingParentCount`、`childrenMap`、`deliverReadyChildren`、`waittingJobs` 和网络 publish。
- 暂时保留 `--mode legacy` 作为 A/B 对照，验收后再删除。

### Phase 3：ns-3 worker 模式

- 订阅 `orchestrator/network/dispatch`，解析 JSON batch。
- 每个 transfer 独立关联 `event_id/transfer_id`，按真实 ns-3 收包完成时间上报。
- 删除 synthetic max-delay 聚合完成路径；同批多个流不能以最大时延一次性全部完成。
- 保留 ns-3 内部路由、队列、链路争用和集合通信流展开能力。

### Phase 4：切换控制权

- 启动进程数由 2 个联邦成员改为 3 个：orchestrator、GPUSim、ns-3。
- 输入转换器输出统一 `workflow.json`，不再输出让 GPUSim 自行控制的 jobs DAG。
- orchestrator 成为唯一的 end 事件发布者。
- legacy 模式只用于回归，不用于正式实验。

### Phase 5：验证

- 状态机测试：链、菱形、fan-in/fan-out、跨 host、多包、失败重试、重复消息。
- 因果测试：同一时间多 publish、零本地事件、完成早于其他 simulator next event、长空闲期。
- 网络测试：并发流共享瓶颈时分别完成，不能使用 `max(delay)` 合并完成。
- 端到端测试：现有 2-rank smoke case 和 4-rank ATLAHS 转换 case 任务数、网络事件数全部闭合。
- 守恒检查：`dispatched = completed + failed + canceled + inflight`；DAG 完成时 inflight 必须为 0。

## 8. 验收标准

- GPUSim worker 启动参数中不再需要完整 DAG 文件。
- GPUSim 源码的 worker 路径不包含后继任务解锁和网络任务生成。
- ns-3 不读取任务依赖，只处理 transfer/collective 命令。
- orchestrator 事件日志可独立重建每个 task 和 edge 的最终状态。
- 任意两个相同时间完成的网络流都有独立 completion，且只发生一次。
- 更换 GPUSim 或 ns-3 为 fake worker 时，DAG 控制测试仍可运行。
- broker 不包含 `task_id`、DAG、compute、network 等领域判断。

## 9. 主要风险

- 当前 GPUSim 的静态 jobs 初始化路径较深，动态创建任务需要隔离 ID 分配和 host 绑定。
- 当前 ns-3 同时存在真实发包和 synthetic completion，必须在迁移中选定唯一完成来源，否则会重复完成。
- FNCS 2.3.2 的消息缓存是按 key 管理，必须使用 JSON batch 和事件 ID，不能依赖最后一个字符串值表达多事件。
- 现有输入转换器对集合通信和 ATLAHS trace 做了聚合，控制面迁移不会自动提高模型准确度；准确度工作需要在新架构稳定后单独验证。
- 迁移期间 legacy/worker 双模式会增加短期复杂度，必须设定删除 legacy 的验收门槛。
