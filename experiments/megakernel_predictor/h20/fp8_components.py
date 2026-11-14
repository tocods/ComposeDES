#!/usr/bin/env python3
"""Standalone resident FP8 service calibration with real routed activations."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import torch

from fp8_full import Checkpoint, DEVICE, alloc_scratchpad, mono_moe, jit_moe, AutoConfig


def measure(fn, iterations=80):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    refs = []
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        refs.append(start.elapsed_time(end) * 1000 / iterations)
    return sorted(refs)[2]


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.tiled.resolve()))
    from monomoe_h20 import activation, specialize

    flavor = os.environ['QWEN_MONO_BOUNDARY']
    specialize(jit_moe._get_monomoe_csrc_dir(), args.tiled / f'upstream-monomoe-{flavor}-grid64', 64)
    cfg = AutoConfig.from_pretrained(args.model).text_config
    scratch = alloc_scratchpad(DEVICE)
    with ExitStack() as stack:
        ckpt = Checkpoint(args.model, stack)
        for layer in args.layers:
            weights = ckpt.experts(layer, cfg)
            for index in args.forwards:
                pack, logits, frame = activation(args.capture, 2, index * 40 + layer)
                if frame[0] != layer:
                    raise ValueError(f'unexpected frame {frame}')
                x = pack.hidden_states_q
                y = torch.empty_like(x)
                fn = lambda: mono_moe(x, logits, *weights, cfg.num_experts_per_tok, out=y,
                                     scratchpad=scratch, interleave_up=False)
                values = measure(fn)
                if not y.isfinite().all().item():
                    raise RuntimeError(f'non-finite output for {frame}')
                print(json.dumps(dict(flavor=flavor, layer=layer, frame=frame, us=values,
                                      unique=pack.topk_ids.unique().numel(),
                                      output_norm=y.float().norm().item())), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('capture', type=Path)
    p.add_argument('--tiled', type=Path, required=True)
    p.add_argument('--layers', nargs='+', type=int, required=True)
    p.add_argument('--forwards', nargs='+', type=int, default=[0, 1, 2, 3])
    run(p.parse_args())
