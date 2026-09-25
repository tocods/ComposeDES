# FNCS Workflow Orchestrator

The orchestrator is the workflow control-plane federate for co-simulation v2. It owns DAG dependency state, dispatches compute and network commands, and decides global completion. FNCS broker remains a generic pub/sub and conservative-time coordinator.

## Run

Start a broker for three federates, then start ns-3 and GPUSim with their `fncs.worker.zpl` files. Start the orchestrator with:

```bash
FNCS_CONFIG_FILE=fncs/orchestrator/fncs.zpl \
COSIM_RUN_ID=my-run \
python3 fncs/orchestrator/workflow_orchestrator.py jobs.json \
  --active-dependency-coordination \
  --event-log output/control-plane.jsonl
```

The workflow input may be the legacy jobs array or an object with `tasks` and
`collectives`. In worker mode GPUSim does not read that file; each task
specification is sent in `compute.dispatch` only when its dependencies are
ready. The orchestrator currently expands ring `allreduce`, `allgather`,
`reduce_scatter`, and `alltoall` operations into point-to-point ns-3 flows and
enforces a barrier between ring steps.

`bytes_per_rank` is the rank's input buffer size. Ring AllReduce and
ReduceScatter send `ceil(bytes_per_rank / ranks)` per rank per step; AllGather
sends one rank input buffer per step; AllToAll sends
`ceil(bytes_per_rank / ranks)` to each non-local peer.

```json
{
  "tasks": [],
  "collectives": [
    {
      "collective_id": "allreduce0",
      "type": "allreduce",
      "algorithm": "ring",
      "bytes_per_rank": 120000,
      "participants": [
        {"src_task_id": "pre0", "dst_task_id": "post0", "host": "host1"},
        {"src_task_id": "pre1", "dst_task_id": "post1", "host": "host2"}
      ]
    }
  ]
}
```

Failed compute and network attempts can be retried by passing
`--max-compute-retries N` and `--max-network-retries N`. A task may override
the compute default with its own `max_retries` field. Every worker event must
carry a unique `event_id`; duplicate events and dispatches are idempotent.

## Coordination and acceleration

With `FNCS_ACTIVE_DEPENDENCY=yes`, the broker accepts atomic dependency graph
updates from the orchestrator on the reserved `__fncs/active_dependencies`
topic. A federate can advance to the transitive lower bound of only the
federates that can currently affect it; the broker falls back to conservative
global-min scheduling if no usable graph has been published. Set
`FNCS_COORDINATION_METRICS=/path/metrics.json` to record grants and rounds.

ComposeDES describes these mechanisms along two optimization axes:

- **Vertical optimization** contracts certified local event chains inside one
  federate. It reduces the number of compute events, dispatches, and grants.
- **Horizontal optimization** exposes independent timelines through federate
  partitioning and uses Active-dependency coordination to advance them without
  waiting for unrelated federates.

Federate partitioning exposes concurrency; Active-dependency exploits that
concurrency. They belong to the same horizontal axis but remain separate
architecture and scheduling decisions. A controlled Active-dependency ablation
must therefore hold the partition fixed.

The orchestrator publishes a new dependency epoch only when a worker event
changes one of three relevant states: compute in flight, network in flight, or
terminal status. The broker translates federate names to integer indexes and
builds the transitive dependency closure only when the topology changes.
Normal scheduling rounds and frontier-only epochs reuse indexed state, closure,
and scratch buffers. Invalid or cyclic graphs retain the conservative
global-minimum fallback.

When only a certified frontier changes, the orchestrator sends a
`frontier_only=1` epoch. The broker retains the indexed dependency graph and
updates only the frontier vector, avoiding a repeated closure rebuild on every
completion certificate. A topology change or frontier withdrawal still sends
a full atomic graph update.

The dispatch path also precompiles immutable data before the FNCS session
starts. Each task's children-free dispatch description is deep-copied once at
workflow construction, and the batch encoder serializes the schema and run-id
prefix once per run. Runtime encoding only fills the batch sequence, logical
time, microstep, and dynamic event array. When an event log is enabled, the
orchestrator writes that structured batch directly instead of decoding the JSON
it just published. Dynamic completion times, retries, frontiers, and failure
states remain runtime data, so causality and Active-dependency behavior are
unchanged.

With `--safe-frontier-coordination`, the orchestrator derives a conservative
earliest completion time from every in-flight compute and network command and
publishes the minimum as `frontier.<federate>`. A frontier certifies that the
named consumer cannot receive a new dispatch before that time. The broker may
therefore grant an idle consumer up to the frontier even when its controller
dependency has an earlier request. The certificate is local to the consumer
and is never propagated as that consumer's producer bound.

Active mode also has an experimental conservative asynchronous grant fast path.
It is disabled by default; enable it only with
`FNCS_ASYNCHRONOUS_GRANTS=yes` after validating the workload. Once a
consumer has submitted a request, the broker can grant it immediately when
every federate in its transitive producer closure is already at a safe request,
or when the consumer's frontier certifies that no earlier input can arrive.
Unrelated federates do not hold that grant behind the global barrier. If the
condition cannot be proved, scheduling falls back to the normal conservative
round. The coordination metrics include `asynchronous_grants`; this path
reduces barrier waiting and scheduler rounds without changing grant counts or
event causality.

In this mode workers and the orchestrator request the maximum FNCS time as a
cancellable lease. Active dependencies still wake the orchestrator at the
first producer completion, while frontiers let unrelated idle workers skip
short polling rounds. Compute certificates subtract the configurable
`--frontier-compute-safety-margin-ns` (10 microseconds by default) to cover the
GPUSim/CloudSim floating-point time conversion. If any in-flight command lacks
a lower bound, the orchestrator withdraws all explicit frontiers and the
broker retains the dependency-safe rule.

An object workflow may contain an `acceleration` section. Regions are eligible
for contraction only when all four closure declarations (`causal`, `state`,
`temporal`, and `resource_non_interference`) are true, the members form a
single-host linear chain, and the region contains no network or collective
boundary. Exact regions must match the detailed duration exactly. Approximate
regions carry lower/upper duration bounds; the optimizer refines only regions
whose upper path can still affect the critical path until the requested
`error_budget_ns` is met. `--optimization-report` writes the certificate and
the selected refinements.

Backend-local chains are disabled by default. Enable them with
`--backend-local-chains` or `COSIM_BACKEND_LOCAL_CHAINS=yes`; limit one plan
with `--backend-local-chain-max-tasks N` or
`COSIM_BACKEND_LOCAL_CHAIN_MAX_TASKS` (default 32, 0 means unlimited). The
coordinator only creates a plan for a same-host, zero-network, single-entry
linear chain when every task using that host is statically ordered. Collective
participants and tasks with retries are excluded. GPUSim still executes every
original `GpuJob` in order and returns every task's original start/finish
timestamp in one boundary batch, so this option removes coordination traffic
without replacing the compute model or changing the boundary trace.

## Test

```bash
python3 -m unittest discover -s fncs/orchestrator/tests -v
tests/active_dependency_smoke/run.sh
tests/control_plane_alpha4_small/run.sh
python3 tests/backend_local_chain_gpu_compare.py
```
