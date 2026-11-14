#!/usr/bin/env python3
"""Measure an H20 MonoMoE specialization against a same-weight Frost control."""
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import torch
from flashinfer.fused_moe import (
    BackendOptions, ExpertConfig, FrostLowLatencyConfig, MoEActivationPack, MoEConfig,
    MoELayer, MoEWeightPack, QuantConfig, QuantFormat, RoutingConfig, alloc_scratchpad,
    interleave_for_tma_wgmma_up, mono_moe,
)
from flashinfer.jit import monomoe as jit_moe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tuning'))
from frost_smoke import E, H, I, K, latency, reference


def specialize(source, dest, grid):
    shutil.copytree(source, dest, dirs_exist_ok=True)
    header = dest / 'src/moe_interface.h'
    old = 'static constexpr std::uint32_t GRID_SIZE = 128;'
    text = header.read_text()
    if text.count(old) != 1 or grid not in (32, 48, 64):
        raise ValueError('expected upstream single-shape header and a grid divisible by 16')
    header.write_text(text.replace(old, f'static constexpr std::uint32_t GRID_SIZE = {grid};'))
    flavor = os.getenv('QWEN_MONO_BOUNDARY', 'ready')
    if flavor == 'barrier':
        path = dest / 'src/moe.cuh'
        text = path.read_text()
        entry = '  moe_down_projection_allexperts_wgmma_tma<Dims>('
        if text.count(entry) != 1:
            raise RuntimeError('unexpected MonoMoE up-to-down handoff')
        text = text.replace(entry, '  cooperative_groups::this_grid().sync();\n' + entry)
        path.write_text('#include <cooperative_groups.h>\n' + text)
    if flavor in ('split', 'split_cached'):
        path = dest / 'src/moe.cuh'
        text = path.read_text().replace('template <typename Dims>\n__device__ void moe_kernel_topk_impl',
                                       'template <typename Dims, int Phase>\n__device__ void moe_kernel_topk_impl')
        text = text.replace('template <typename Dims>\n__global__ __launch_bounds__',
                            'template <typename Dims, int Phase = 0>\n__global__ __launch_bounds__')
        text = text.replace('moe_kernel_topk_impl<Dims>(', 'moe_kernel_topk_impl<Dims, Phase>(')
        text = text.replace('  {\n    const uint32_t partial_n', '  if constexpr (Phase != 2) {\n    const uint32_t partial_n')
        text = text.replace('    shmem->scale_parity = flip & 1u;\n    spec->launch_flip[blockIdx.x] = flip + 1u;',
                            '    shmem->scale_parity = (flip - (Phase == 2)) & 1u;\n'
                            '    if constexpr (Phase != 2) spec->launch_flip[blockIdx.x] = flip + 1u;')
        text = text.replace('  {\n    constexpr uint32_t SCALE_N', '  if constexpr (Phase != 2) {\n    constexpr uint32_t SCALE_N')
        text = text.replace('  if (in_up && up_group < shmem->expert_count) {',
                            '  if constexpr (Phase != 2) if (in_up && up_group < shmem->expert_count) {')
        text = text.replace('  // ── Phase 4: down-projection',
                            '  if constexpr (Phase == 1) return;\n\n  // ── Phase 4: down-projection')
        if flavor == 'split_cached':
            header = dest / 'src/moe_internal.h'
            tail = header.read_text()
            anchor = '  uint32_t launch_flip[Dims::KernelConfig::GRID_SIZE];'
            tail = tail.replace(anchor, anchor + '\n  uint16_t cut_slot[Dims::NUM_EXPERTS];'
                                '\n  uint8_t cut_count[Dims::NUM_EXPERTS];'
                                '\n  uint8_t cut_rank[Dims::NUM_EXPERTS][Dims::BS];'
                                '\n  ExpertRef cut_experts[Dims::NUM_EXPERTS];\n  uint32_t cut_used;')
            header.write_text(tail)
            patch = Path(__file__).with_name('monomoe-cut-metadata.inc').read_text()
            save, restore = patch.split('\n@@restore\n')
            text = text.replace('  const unsigned warp_id = get_any_warp<Dims>();',
                                '  const unsigned warp_id = get_any_warp<Dims>();\n  if constexpr (Phase != 2) {')
            text = text.replace('  // ── Phase 3: up-projection', save + '\n' + restore + '\n  // ── Phase 3: up-projection')
        path.write_text(text)
    if flavor in ('flags', 'barrier', 'split', 'split_cached'):
        wrapper = dest / 'monomoe_wrapper.cuh'
        text = wrapper.read_text()
        start = text.index('  moe_kernel_topk<Dims><<<')
        end = text.index('  CUDA_CHECK(cudaGetLastError());', start)
        launch = Path(__file__).with_name('monomoe-launch.inc').read_text()
        if flavor in ('split', 'split_cached'):
            cut = launch.index('  CUDA_CHECK(cudaLaunchCooperativeKernel(')
            launch = launch[:cut] + ''.join(launch[cut:].replace('moe_kernel_topk<Dims>', f'moe_kernel_topk<Dims, {phase}>')
                                          for phase in (1, 2))
            text = text[:start] + launch + text[end:]
            attr = text.index('  CUDA_CHECK(cudaFuncSetAttribute(moe_kernel_topk<Dims>,')
            attr_end = text.index('\n  //', attr)
            init = text[attr:attr_end]
            text = text[:attr] + ''.join(init.replace('moe_kernel_topk<Dims>', f'moe_kernel_topk<Dims, {phase}>') + '\n'
                                         for phase in (1, 2)) + text[attr_end:]
        else:
            text = text[:start] + launch + text[end:]
        wrapper.write_text(text)
    jit_moe._get_monomoe_csrc_dir = lambda: dest
    jit_moe.get_monomoe_uri = lambda: f'monomoe_h20_{flavor}_grid{grid}'


def weights(rows, cols):
    raw = torch.randn(E, rows, cols, device='cuda', dtype=torch.bfloat16)
    w = (raw * 32).to(torch.float8_e4m3fn)
    scale = torch.rand(E, rows // 128, cols // 128, device='cuda').mul_(0.001).add_(0.0007)
    return w, scale


def activation(path, batch, index):
    stride = 16 + 4 * batch * (2 * H + E + K)
    with path.open('rb') as file:
        file.seek(index * stride)
        raw = file.read(stride)
    layer, actual, cohort, step = struct.unpack_from('<4i', raw)
    if actual != batch:
        raise ValueError(f'incorrect capture batch in {path}')
    array = np.frombuffer(raw, '<f4', count=batch * (H + E), offset=16).copy()
    x = torch.from_numpy(array[:batch * H].reshape(batch, H)).to('cuda', torch.bfloat16)
    logits = torch.from_numpy(array[batch * H:].reshape(batch, E)).to('cuda', torch.bfloat16)
    ids = torch.argsort(logits.float(), descending=True, stable=True)[:, :K]
    values = logits.float().gather(1, ids)
    pack = MoEActivationPack(x, None, ids.int().contiguous(), values.softmax(1).contiguous())
    return pack, logits, (layer, cohort, step)


def mono_reference(pack, w1, s1, w2, s2):
    def round_blocks(x, width):
        tiles = x.float().reshape(-1, width)
        maximum = tiles.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        return ((tiles * (448 / maximum)).to(torch.float8_e4m3fn).float() * (maximum / 448)).reshape_as(x)

    def unpack(weight, scale):
        return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)

    out = []
    for x, ids, scores in zip(pack.hidden_states_q, pack.topk_ids.tolist(), pack.topk_weights):
        x = round_blocks(x, 128)
        total = torch.zeros(H, device=x.device)
        for expert, score in zip(ids, scores):
            up, gate = (unpack(w1[expert], s1[expert]) @ x).chunk(2)
            q = round_blocks(up * torch.nn.functional.silu(gate) * score, 64)
            total += unpack(w2[expert], s2[expert]) @ q
        out.append(total.bfloat16())
    return torch.stack(out)


def main():
    grid = int(sys.argv[1])
    flavor = os.getenv('QWEN_MONO_BOUNDARY', 'ready')
    parent = Path(__file__).resolve().parent
    specialize(jit_moe._get_monomoe_csrc_dir(), parent / f'upstream-monomoe-{flavor}-grid{grid}', grid)
    device = torch.device('cuda:0')
    if torch.cuda.get_device_properties(device).multi_processor_count < grid:
        raise RuntimeError('too few SMs for the upstream residency contract')
    torch.manual_seed(7162)
    w1, s1 = weights(2 * I, H)  # Frost canonical [up, gate].
    w2, s2 = weights(H, I)
    mono_w1 = interleave_for_tma_wgmma_up(torch.cat([w1[:, I:], w1[:, :I]], dim=1))
    mono_s1 = torch.cat([s1[:, I // 128:], s1[:, :I // 128]], dim=1).contiguous()
    config = MoEConfig(routing=RoutingConfig(E, K), experts=ExpertConfig(I, local_num_experts=E),
                       quant=QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
                       backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)))
    frost = MoELayer(config, device=device)
    fpack = MoEWeightPack()
    fpack.prepare_for('frost_low_latency', FrostLowLatencyConfig.prepare_deepseek_fp8_weights(w1, s1, w2, s2))
    scratch = alloc_scratchpad(device)
    for b in (2, 4, 8):
        for index in (20, 84, 135):
            p, logits, frame = activation(parent / f'held-b{b}.bin', b, index)
            out_buf = torch.empty_like(p.hidden_states_q)
            out_mono, mono = latency(lambda: mono_moe(p.hidden_states_q, logits, mono_w1, mono_s1, w2, s2,
                                           K, out=out_buf, scratchpad=scratch, interleave_up=False))
            out_frost, control = latency(lambda: frost(p, fpack))
            targets = {'frost': reference(p, w1, s1, w2, s2).float(),
                       'mono': mono_reference(p, w1, s1, w2, s2).float()}
            delta = {name: ((result.float() - targets[name]).norm() / targets[name].norm()).item()
                     for name, result in (('frost', out_frost), ('mono', out_mono))}
            print(json.dumps(dict(grid=grid, flavor=flavor, batch=b, frame=frame, relative_l2=delta,
                                  mono_us=mono, frost_us=control, unique=p.topk_ids.unique().numel())), flush=True)
            if not (0 <= delta['frost'] < .01 and 0 <= delta['mono'] < .035):
                raise ValueError(f'reference check failed at {frame}: {delta}')


if __name__ == '__main__':
    main()
