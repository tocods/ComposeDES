"""Advance two CTA groups through the ComposeDES DAG under measured shared service."""
from collections import defaultdict
from heapq import heappop, heappush
import math


def schedule(tasks, partitions, coefficients):
    """Integrate each active stage's remaining service up to the next event.

    Tile readiness and worker FIFO edges come from the ComposeDES workflow.
    Service is expressed in solo-equivalent ns; all active tiles in a role
    share its virtual clock. Zero-duration dependency nodes drain at one time.
    """
    indexed = {task['name']: task for task in tasks}
    indegree = dict.fromkeys(indexed, 0)
    for task in tasks:
        for child in task['children']:
            indegree[child['child']] += 1
    ready = [name for name, degree in indegree.items() if not degree]
    heaps = {role: [] for role in ('gate', 'down', 'fixed')}
    virtual = dict.fromkeys(heaps, 0.)
    solo = dict(zip(('gate', 'down'), partitions))
    actual = {}
    overlap_ns = 0.
    now = 0.
    while ready or any(heaps.values()):
        for name in ready:
            task = indexed[name]
            role = task.get('stage', 'fixed')
            actual[name] = now
            heappush(heaps[role], (virtual[role] + task['estimated_duration_ns'], name))
        ready.clear()
        speeds = {'fixed': 1.}
        for role, opposite in (('gate', 'down'), ('down', 'gate')):
            pressure = len(heaps[opposite]) / solo[opposite]
            slowdown = 1 + coefficients[role]['opposite'] * pressure
            speeds[role] = 1 / slowdown
        dt = min(max(0, heap[0][0] - virtual[role]) / speeds[role]
                 for role, heap in heaps.items() if heap)
        if heaps['gate'] and heaps['down']:
            overlap_ns += dt
        now += dt
        for role, heap in heaps.items():
            if heap:
                virtual[role] += speeds[role] * dt
            while heap and heap[0][0] <= virtual[role] + 1e-6:
                _, name = heappop(heap)
                actual[name] = now - actual[name]
                for child in indexed[name]['children']:
                    target = child['child']
                    indegree[target] -= 1
                    if not indegree[target]:
                        ready.append(target)
    if len(actual) != len(tasks) or not math.isfinite(now):
        raise ValueError('incomplete event schedule')
    return now, overlap_ns, actual
