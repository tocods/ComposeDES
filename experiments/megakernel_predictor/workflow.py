"""Translate a resident MoE execution plan into a ComposeDES workflow DAG."""
from dataclasses import dataclass
from heapq import heappop, heappush
from functools import lru_cache
from pathlib import Path
import sys


ORCHESTRATOR = Path(__file__).resolve().parents[2] / 'fncs' / 'orchestrator'
if not (ORCHESTRATOR / 'graph_optimizer.py').is_file():
    raise SystemExit('initialize the ComposeDES fncs submodule with git submodule update --init fncs')
sys.path.insert(0, str(ORCHESTRATOR))
from graph_optimizer import longest_path_certificate, optimize_workflow
from workflow_orchestrator import COMPUTE_COMPLETED, COMPUTE_DISPATCH, WorkflowController

from service import work


@dataclass(frozen=True)
class Candidate:
    mode: int
    producers: int
    gate_rows: int
    down_rows: int
    gate_threads: int
    down_threads: int
    matched: int

    def partition(self, sm):
        if self.mode not in (7, 8, 9):
            raise ValueError(f'unsupported measured kernel mode {self.mode}')
        divided = self.mode == 8 or (self.mode == 7 and self.producers) or self.matched
        if divided:
            if not 0 < self.producers < sm:
                raise ValueError(f'invalid CTA partition {self.producers}/{sm}')
            return self.producers, sm - self.producers
        return sm, sm


def nanoseconds(us):
    return max(0, round(float(us) * 1000))


class Workflow:
    def __init__(self):
        self.tasks = {}

    def add(self, name, estimate, parents=(), radius=0., **metadata):
        if name in self.tasks:
            raise ValueError(f'duplicate task {name}')
        self.tasks[name] = {
            'name': name, 'host': 'gpu0', 'estimated_duration_ns': nanoseconds(estimate),
            'children': [], **metadata,
        }
        if radius:
            self.tasks[name]['duration_bounds_ns'] = [nanoseconds(estimate * (1 - radius)),
                                                      nanoseconds(estimate * (1 + radius))]
        for parent in dict.fromkeys(parents):
            self.tasks[parent]['children'].append({'child': name, 'size': 0})

    def serialize(self):
        return list(self.tasks.values())


def build(service, route, candidate, sm, policy, uncertainty=False):
    first, second = candidate.partition(sm)
    gt, ge = service.stage(route.layer, route.loads, 'gate', first, candidate.gate_threads, candidate.gate_rows)
    dt, de = service.stage(route.layer, route.loads, 'down', second, candidate.down_threads, candidate.down_rows)
    plain, cooperative, barrier, setup, reduce = service.overhead[route.layer]
    sync = max(0, barrier - cooperative)
    wf = Workflow()
    wf.add('launch', cooperative)
    wf.add('setup', max(0, setup - cooperative) + sync, ('launch',))
    if candidate.mode != 8 or policy == 'additive':
        wf.add('gate', gt, ('setup',), ge if uncertainty else 0)
        boundary = cooperative if candidate.mode == 9 else sync if candidate.mode == 7 else 0
        wf.add('boundary', boundary, ('gate',))
        wf.add('down', dt, ('boundary',), de if uncertainty else 0)
        tails = ['down']
    else:
        ready = []
        for name, blocks, rows, estimate, radius in (
            ('gate', first, candidate.gate_rows, gt, ge),
            ('down', second, candidate.down_rows, dt, de),
        ):
            tiles, cta = work(route.loads, name, blocks, rows)
            unit = estimate / max(cta)
            previous = ['setup'] * blocks
            for expert, demand in enumerate(route.loads):
                completions = []
                for tile in range(tiles):
                    index = expert * tiles + tile
                    worker = index % blocks
                    node = f'{name}/{index}'
                    parents = [previous[worker]] + ([ready[expert]] if name == 'down' else [])
                    wf.add(node, unit * demand, parents, radius if uncertainty else 0,
                           stage=name, cta=worker, expert=expert)
                    previous[worker] = node
                    completions.append(node)
                if name == 'gate':
                    node = f'ready/{expert}'
                    wf.add(node, 0., completions)
                    ready.append(node)
            if name == 'gate':
                upstream_tails = previous
        tails = upstream_tails + previous
    wf.add('reduce', max(0, reduce - plain) + sync, tails)
    return wf.serialize()


@lru_cache(maxsize=1024)
def score_layer(service, route, candidate, sm, policy, uncertainty=False):
    tasks = build(service, route, candidate, sm, policy, uncertainty)
    bounds = longest_path_certificate(tasks)
    if uncertainty:
        for task in tasks:
            task.pop('duration_bounds_ns', None)
        nominal = longest_path_certificate(tasks)['lower_ns']
    else:
        nominal = bounds['lower_ns']
    return nominal, bounds['lower_ns'], bounds['upper_ns']


def replay(tasks):
    """Run a fixed-service local worker against the same FNCS controller used by the federation."""
    controller = WorkflowController(tasks, 'resident-moe-replay')
    events = []

    def dispatch(commands, now):
        for command in commands:
            if command.topic != COMPUTE_DISPATCH:
                raise ValueError(f'unsupported worker command {command.topic}')
            payload = command.event['payload']
            done = now + payload['task']['estimated_duration_ns']
            completion = {'task_id': payload['task_id'], 'attempt': payload['attempt'], 'status': 'SUCCEEDED'}
            heappush(events, (done, payload['task_id'], command.event['correlation_id'], completion))

    now = 0
    dispatch(controller.initial_commands(), now)
    while events:
        now, name, correlation, completion = heappop(events)
        event = {'event_id': f'done/{name}', 'kind': 'compute.completed',
                 'correlation_id': correlation, 'payload': completion}
        dispatch(controller.handle_event(COMPUTE_COMPLETED, event), now)
    if not controller.done:
        raise RuntimeError('ComposeDES workflow did not finish')
    return now


def contract_prologue(tasks):
    # Neither task publishes to any tile before setup closes. This region
    # has one host and fixed service, and starts before either CTA role uses it.
    closure = dict(causal=True, state=True, temporal=True, resource_non_interference=True)
    macro_task = {'name': 'prologue', 'host': 'gpu0'}
    region = {'id': 'prologue', 'members': ['launch', 'setup'], 'mode': 'exact',
              'closure': closure, 'macro_task': macro_task}
    return optimize_workflow(tasks, [], {'regions': [region], 'error_budget_ns': 0})
