"""ComposeDES worker with GPU kernel admission and fluid shared-service rates."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

from joint_cases import COMPUTE, DEPENDENCIES, TOTAL, downstream

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT))
from workflow import Workflow, longest_path_certificate
from workflow_orchestrator import COMPUTE_COMPLETED, WorkflowController


@dataclass(frozen=True)
class Admission:
    order_weight: float
    free_sm: int
    sm_contention: float
    memory_contention: float


# The grid is shared across tasks, shapes, layouts, and graphs. No target node
# timing or per-graph fit enters the zero-target-measurement predictions.
PARAMETERS = tuple(Admission(*values) for values in product(
    (0., .5, 1.), (24, 64, 96), (.75, 2., 4.), (0., .15)
))


def specs(dependencies, order, work):
    graph = Workflow()
    for i in order:
        # Workflow stores estimated work in ns and accepts duration in us.
        graph.add(str(i), work[i]*1000, map(str, dependencies[i]))
    return graph.serialize()


@lru_cache(maxsize=1024)
def dag_only(family, work, order):
    return longest_path_certificate(specs(DEPENDENCIES[family], order, work))["lower_ns"]/1e6


@dataclass(frozen=True)
class Candidate:
    cap: str
    order: tuple[int, ...]
    work: tuple[float, ...]


def replay(deps, candidate, model, sm_count=108):
    """Replay one graph. WorkflowController releases only causally ready tasks.

    SM transfer kernels consume CTA slots, whereas DMA occupies a copy engine.
    Transfers share a fluid host-memory rate; simultaneous compute can slow it.
    On an event, reevaluate admissions and advance all active work to the next
    completion. Creation rank is a bounded priority input and imposes no edges.
    """
    order, cap, work = candidate.order, candidate.cap, candidate.work
    workflow = WorkflowController(specs(deps, order, work), "gpu-graph-admission")
    tails = downstream(deps, work)
    normalizer = max(tails)
    ranks = {n: i/(TOTAL-1) for i, n in enumerate(order)}
    priority = {
        n: model.order_weight * (1-ranks[n]) + (1-model.order_weight) * tails[n]/normalizer
        for n in order
    }
    copy_sm = 0 if cap == "dma" else int(cap)
    compute_sm = sm_count - model.free_sm
    ready = {int(c.event["payload"]["task_id"]): c for c in workflow.initial_commands()}
    active = {}
    now = 0.

    while ready or active:
        for node in sorted(ready, key=lambda n: (-priority[n], n)):
            active_compute = sum(n < COMPUTE for n in active)
            active_copies = len(active) - active_compute
            sm_used = active_compute*compute_sm + active_copies*copy_sm
            if node < COMPUTE:
                admitted = not active_compute and sm_used+compute_sm <= sm_count
            elif cap == "dma":
                admitted = not active_copies
            else:
                admitted = sm_used+copy_sm <= sm_count
            if admitted:
                active[node] = [work[node], ready.pop(node)]
        if not active:
            raise RuntimeError("No work admitted despite a nonempty ready queue")

        compute_running = any(n < COMPUTE for n in active)
        copy_count = sum(n >= COMPUTE for n in active)
        sm_pressure = copy_count * copy_sm/64
        memory_penalty = model.memory_contention
        rates = {
            node: (
                1/(1 + model.sm_contention*sm_pressure + memory_penalty*copy_count)
                if node < COMPUTE else
                1/(copy_count*(1 + memory_penalty*compute_running))
            )
            for node in active
        }
        duration = min(remaining/rates[node] for node, (remaining, _) in active.items())
        now += duration
        finished = []
        for node, entry in active.items():
            entry[0] -= duration*rates[node]
            if entry[0] < 1e-8:
                finished.append(node)
        for node in finished:
            _, command = active.pop(node)
            completion = {
                "event_id": f"done/{node}",
                "kind": "compute.completed",
                "correlation_id": command.event["correlation_id"],
                "payload": {"task_id": str(node),
                            "attempt": command.event["payload"]["attempt"], "status": "SUCCEEDED"},
            }
            commands = workflow.handle_event(COMPUTE_COMPLETED, completion)
            ready.update((int(c.event["payload"]["task_id"]), c) for c in commands)
    if not workflow.done:
        raise RuntimeError("ComposeDES did not complete the workload")
    return now
