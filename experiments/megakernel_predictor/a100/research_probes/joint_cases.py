"""Frozen task topologies, candidate space, and leave-out rules."""
from itertools import product
from random import Random

COPIES = 6
COMPUTE = 7
TOTAL = COPIES + COMPUTE
CAPS = ("dma", "2", "8", "24", "64")
ORDERS = ("io", "compute", "critical", "alternating", "random-7", "random-19")
TRAIN_ORDERS = ORDERS[:4]
SHAPES = {"attention": (128, 512, 1536), "gemm": (1024, 3072, 8192)}
FAMILIES = ("chain", "gated", "join")
LAYOUTS = ("packed", "paged")
TOKENS = 2048
PAGE = 256

# Tasks C0..C6 are 0..6. Tasks H0..H5 are 7..12.
# H depends on C in the latter two graphs because the preceding work releases
# a known readback plan. The selected indices are fixed when replaying a plan.
DEPENDENCIES = {
    "chain": ((), (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
              (), (7,), (8,), (9,), (10,), (11,)),
    "gated": ((), (0, 7), (1, 8), (2, 9), (10,), (4, 11), (3, 5, 12),
              (0,), (1,), (2,), (), (10,), (11,)),
    "join": ((), (0, 7), (1, 8), (9,), (3, 10), (11,), (2, 4, 5, 12),
             (0,), (7,), (), (3,), (), (11,)),
}

WORK = {
    "chain": (1, 1, 1, 1, 1, 1, 1),
    "gated": (3, 2, 1, 1, 1, 1, 1),
    "join": (2, 1, 3, 1, 1, 1, 1),
}


def jobs():
    for family, (kernel, shapes), layout in product(FAMILIES, SHAPES.items(), LAYOUTS):
        for shape in shapes:
            yield family, kernel, shape, layout


def downstream(deps, work):
    children = [[i for i, ps in enumerate(deps) if node in ps] for node in range(len(deps))]
    memo = {}

    def tail(i):
        if i not in memo:
            memo[i] = work[i] + max((tail(child) for child in children[i]), default=0)
        return memo[i]

    return tuple(tail(i) for i in range(len(deps)))


def create_orders(family, isolated):
    deps = DEPENDENCIES[family]
    tails = downstream(deps, isolated)
    output = {}
    for label in ORDERS:
        rng = Random(label)
        order = []
        while len(order) < TOTAL:
            ready = [i for i, parents in enumerate(deps)
                     if i not in order and all(p in order for p in parents)]
            if label == "io":
                pick = min(ready, key=lambda i: (i < COMPUTE, i))
            elif label == "compute":
                pick = min(ready, key=lambda i: (i >= COMPUTE, i))
            elif label == "critical":
                pick = max(ready, key=lambda i: (tails[i], -i))
            elif label == "alternating":
                previous_was_compute = not order or order[-1] < COMPUTE
                pick = min(ready, key=lambda i: ((i < COMPUTE) == previous_was_compute, i))
            else:
                pick = rng.choice(ready)
            order.append(pick)
        output[label] = tuple(order)
    return output
