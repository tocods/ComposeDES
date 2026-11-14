#!/usr/bin/env python3
"""Check FlashInfer's routed FP8 operators on the Qwen3.5-35B shape."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

from flashinfer.fused_moe import (
    BackendOptions, CutlassFp8BlockConfig, ExpertConfig, FrostLowLatencyConfig,
    MoEActivationPack, MoEConfig, MoELayer, MoEWeightPack, QuantConfig, QuantFormat,
    RoutingConfig,
)


E, H, I, K = 256, 2048, 512, 8


def read_input(path, batch):
    # Capture header followed by activation, logits, top-k IDs, and expert output.
    raw = path.read_bytes()
    stride = 16 + 4 * batch * (2 * H + E + K)
    frame = raw[20 * stride:21 * stride]
    layer, recorded_batch, cohort, step = struct.unpack_from('<4i', frame)
    if batch != recorded_batch:
        raise ValueError(f'unexpected batch in {path}')
    x = np.frombuffer(frame, '<f4', count=batch * H, offset=16).reshape(batch, H)
    logits = np.frombuffer(frame, '<f4', count=batch * E, offset=16 + 4 * batch * H).reshape(batch, E)
    ids = np.frombuffer(frame, '<i4', count=batch * K, offset=16 + 4 * batch * (H + E)).reshape(batch, K)
    x = torch.from_numpy(x.copy()).to('cuda', torch.bfloat16)
    logits = torch.from_numpy(logits.copy()).cuda()
    ids = torch.from_numpy(ids.copy()).cuda()
    scores = logits.gather(1, ids.long()).softmax(dim=1)
    return MoEActivationPack(x, None, ids, scores), (layer, cohort, step)


def make_weights(rows, cols):
    # Synthetic weights keep the layout and per-block scales explicit.
    weights = torch.randn(E, rows, cols, device='cuda', dtype=torch.bfloat16)
    weights = weights.mul_(32).to(torch.float8_e4m3fn)
    scales = torch.full((E, rows // 128, cols // 128), 0.001, device='cuda')
    return weights, scales


def latency(fn, repeats=80):
    output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    stop.record()
    stop.synchronize()
    torch.testing.assert_close(output, captured)
    return output, start.elapsed_time(stop) * 1000 / (5 * repeats)


def reference(inputs, w1, s1, w2, s2):
    def block_quantize(x):
        chunks = x.float().reshape(-1, 128)
        scale = chunks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
        return ((chunks / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(x)

    def unpack(weight, scale):
        return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)

    out = []
    for x, ids, scores in zip(inputs.hidden_states_q, inputs.topk_ids.tolist(), inputs.topk_weights):
        x = block_quantize(x)
        total = torch.zeros(H, device=x.device)
        for e, score in zip(ids, scores):
            up, gate = (unpack(w1[e], s1[e]) @ x).chunk(2)
            q = block_quantize(up * torch.nn.functional.silu(gate))
            total += score * (unpack(w2[e], s2[e]) @ q).bfloat16().float()
        out.append(total.bfloat16())
    return torch.stack(out)


def main():
    torch.manual_seed(7162)
    device = torch.device('cuda:0')
    w1, s1 = make_weights(2 * I, H)
    w2, s2 = make_weights(H, I)
    frost = FrostLowLatencyConfig(max_num_tokens=8)
    cutlass = CutlassFp8BlockConfig()
    layers = {}
    compare_cutlass = '--frost-only' not in sys.argv
    for backend in ((frost, cutlass) if compare_cutlass else (frost,)):
        config = MoEConfig(
            routing=RoutingConfig(E, K), experts=ExpertConfig(I, local_num_experts=E),
            quant=QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
            backend=BackendOptions((backend,)),
        )
        layers[type(backend)] = MoELayer(config, device=device)
    fplan = frost.prepare_deepseek_fp8_weights(w1, s1, w2, s2)
    fpack = MoEWeightPack()
    fpack.prepare_for('frost_low_latency', fplan)
    cpack = MoEWeightPack()
    cpack.prepare_for('cutlass_fp8_block', dict(fc1_expert_weights=w1, fc2_expert_weights=w2,
                                              fc1_block_scale=s1, fc2_block_scale=s2))
    for batch in (2, 4, 8):
        inputs, frame = read_input(Path(f'dev/train-b{batch}.bin'), batch)
        out_frost, frost_us = latency(lambda: layers[type(frost)](inputs, fpack))
        baseline = reference(inputs, w1, s1, w2, s2).float()
        row = dict(batch=batch, frame=frame, unique=inputs.topk_ids.unique().numel())

        def validate(name, result, duration):
            error = ((result.float() - baseline).norm() / baseline.norm()).item()
            if not 0 <= error < 0.04:
                raise ValueError(f'{name} output mismatch against reference: {error:.4g}')
            row[name] = dict(latency_us=duration, relative_l2=error)

        validate('frost', out_frost, frost_us)
        if compare_cutlass:
            out_cutlass, cutlass_us = latency(lambda: layers[type(cutlass)](inputs, cpack))
            validate('cutlass', out_cutlass, cutlass_us)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
