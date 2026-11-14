#!/usr/bin/env python3
"""Run the Qwen3.5 text model on H20 with official block-FP8 routed weights."""
import argparse
from contextlib import ExitStack, redirect_stdout
from importlib import import_module
import json
import os
from pathlib import Path
import struct
import sys
from time import perf_counter
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, StaticCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from flashinfer.fused_moe import alloc_scratchpad, interleave_for_tma_wgmma_up, mono_moe
from flashinfer.jit import monomoe as jit_moe


DEVICE = torch.device('cuda:0')


class Checkpoint:
    def __init__(self, root, stack):
        index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.index = index
        self.shards = {shard: stack.enter_context(safe_open(root / shard, framework='pt'))
                       for shard in set(index.values()) if (root / shard).exists()}

    def get(self, key):
        return self.shards[self.index[key]].get_tensor(key).to(DEVICE)

    def dense(self, key):
        weight = self.get(key)
        scales_key = key.removesuffix('.weight') + '.weight_scale_inv'
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.get(scales_key).float().repeat_interleave(128, 0).repeat_interleave(128, 1)
            weight = weight.float() * scale[:weight.shape[0], :weight.shape[1]]
        return weight.bfloat16()

    def experts(self, layer, cfg):
        e, h, i = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        w1 = torch.empty((e, i * 2, h), device=DEVICE, dtype=torch.float8_e4m3fn)
        w2 = torch.empty((e, h, i), device=DEVICE, dtype=torch.float8_e4m3fn)
        s1 = torch.empty((e, i * 2 // 128, h // 128), device=DEVICE, dtype=torch.float32)
        s2 = torch.empty((e, h // 128, i // 128), device=DEVICE, dtype=torch.float32)
        for expert in range(e):
            prefix = f'model.language_model.layers.{layer}.mlp.experts.{expert}'
            for proj, weight, scale in (('gate_proj', w1[expert, :i], s1[expert, :i // 128]),
                                        ('up_proj', w1[expert, i:], s1[expert, i // 128:]),
                                        ('down_proj', w2[expert], s2[expert])):
                weight.copy_(self.get(f'{prefix}.{proj}.weight'))
                scale.copy_(self.get(f'{prefix}.{proj}.weight_scale_inv'))
        return interleave_for_tma_wgmma_up(w1), s1, w2, s2


def routed(self, x):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if x.shape[0] > 8:
        raise ValueError('MonoMoE specialization handles up to 8 tokens per invocation')
    logits = F.linear(x, self.gate.weight)
    profile = getattr(self, 'profile', None)
    if profile is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    out = mono_moe(x, logits, *self.packed, 8, scratchpad=self.scratch, interleave_up=False)
    if profile is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        profile.append((begin, end))
    shared = self.shared_expert(x) * self.shared_expert_gate(x).sigmoid()
    return (shared + out).view(shape)


def load_model(root, scratch):
    cfg = AutoConfig.from_pretrained(root).text_config
    cfg._attn_implementation = 'sdpa'
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5MoeForCausalLM(cfg)
    for layer in model.model.layers:
        del layer.mlp.experts
    with ExitStack() as stack:
        checkpoint = Checkpoint(root, stack)
        for key, param in list(model.named_parameters()):
            source = 'model.language_model.' + key.removeprefix('model.') if key.startswith('model.') else key
            submodule, field = key.rsplit('.', 1)
            tensor = checkpoint.dense(source)
            if tensor.shape != param.shape:
                raise ValueError(f'{key}: {tensor.shape} != {param.shape}')
            setattr(model.get_submodule(submodule), field, nn.Parameter(tensor, requires_grad=False))
        for index, layer in enumerate(model.model.layers):
            block = layer.mlp
            block.packed = checkpoint.experts(index, cfg)
            block.scratch = scratch
            block.forward = MethodType(routed, block)
            print(f'loaded layer {index}', flush=True, file=sys.stderr)
    for key, buffer in model.named_buffers():
        owner, field = key.rsplit('.', 1)
        model.get_submodule(owner)._buffers[field] = buffer.to(DEVICE)
    return model.eval()


def prefill_pair(model, tokenizer, prompts):
    results, lengths = [], []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        cache = None
        lengths.append(tokens.shape[1])
        for chunk in tokens.split(8, dim=1):
            output = model(chunk, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
        results.append(output)
    longest = max(lengths)
    left, right = (output.past_key_values for output in results)
    for a, b in zip(left.layers, right.layers, strict=True):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            if getattr(a, name, None) is None:
                continue
            states = [getattr(a, name), getattr(b, name)]
            if name in ('keys', 'values'):
                states = [F.pad(state, (0, 0, longest - length, 0))
                          for state, length in zip(states, lengths, strict=True)]
            setattr(a, name, torch.cat(states))
    emitted = torch.cat([output.logits[:, -1].argmax(-1) for output in results])
    return emitted, left, lengths


class GraphDecode:
    def __init__(self, model, cache, lengths, teacher, steps, scratch):
        longest = max(lengths)
        self.capacity = longest + steps + 2
        self.cache = StaticCache(config=model.config, max_cache_len=self.capacity)
        self.ids = torch.zeros((2, 1), dtype=torch.long, device=DEVICE)
        self.pos = torch.tensor(lengths, device=DEVICE)[:, None]
        self.columns = torch.arange(self.capacity, device=DEVICE)[None, :]
        self.valid_from = longest - self.pos.clone()
        self.end = torch.tensor(longest, device=DEVICE)
        self.mask = torch.zeros((2, 1, 1, self.capacity), dtype=torch.bool, device=DEVICE)
        snapshots = []
        for dynamic, static in zip(cache.layers, self.cache.layers, strict=True):
            if hasattr(dynamic, 'keys') and dynamic.keys is not None:
                static.lazy_initialization(dynamic.keys, dynamic.values)
                static.keys[:, :, :longest].copy_(dynamic.keys)
                static.values[:, :, :longest].copy_(dynamic.values)
                static.cumulative_length.fill_(longest)
                snapshots.append((static.cumulative_length, static.cumulative_length.clone()))
            else:
                static.lazy_initialization(conv_states=dynamic.conv_states, recurrent_states=dynamic.recurrent_states)
                static.conv_states.copy_(dynamic.conv_states)
                static.recurrent_states.copy_(dynamic.recurrent_states)
                static.has_previous_state = True
                snapshots.extend(((static.conv_states, dynamic.conv_states),
                                  (static.recurrent_states, dynamic.recurrent_states)))
        self.ids.copy_(teacher.reshape(2, 1))
        self.update_mask()

        def forward():
            return model(self.ids, attention_mask=self.mask, position_ids=self.pos,
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits

        self.graph = torch.cuda.CUDAGraph()
        # Restore the complete initial state, including MonoMoE's handoff
        # counters and double-buffer parity, for the first replay.
        snapshots.append((scratch, scratch.clone()))
        torch.cuda.synchronize()
        self.expected = forward().clone()
        self.restore(snapshots)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.logits = forward()
        self.restore(snapshots)

    @staticmethod
    def restore(snapshots):
        for target, source in snapshots:
            target.copy_(source)

    def update_mask(self):
        self.mask[:, 0, 0].copy_((self.columns >= self.valid_from) & (self.columns <= self.end))

    def set_input(self, tokens, position, slot):
        self.ids.copy_(tokens.reshape(2, 1))
        self.pos.copy_(position)
        self.end.fill_(slot)
        self.update_mask()

    def replay(self):
        self.graph.replay()
        return self.logits


@torch.inference_mode()
def run(args, model):
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    prompts = [s.strip() for s in args.prompts.read_text().splitlines() if s.strip()]
    records = []
    outputs = []
    if len(prompts) % 2:
        raise ValueError('B2 requires an even number of prompts')
    teacher = json.loads(args.teacher.read_text()) if args.teacher else None
    if args.teacher_gguf:
        raw = args.teacher_gguf.read_bytes()
        ids = struct.unpack(f'<{len(prompts) * args.steps}i', raw)
        teacher = [[ids[row * args.steps + step * 2:row * args.steps + step * 2 + 2]
                    for step in range(args.steps)] for row in range(0, len(prompts), 2)]
    print('batch,cohort,step,decode_ms,wall_ms,peak_bytes,monomoe_ms', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for cohort, start in enumerate(range(0, len(prompts), 2)):
        emitted, past, lengths = prefill_pair(model, tokenizer, prompts[start:start + 2])
        longest = max(lengths)
        position_start = torch.tensor(lengths, device=DEVICE)[:, None]
        columns = torch.arange(longest + args.steps, device=DEVICE)
        mask = columns[None, :] >= longest - position_start
        outputs.append([])
        graph = None
        if args.cuda_graph:
            initial = emitted if teacher is None else torch.tensor(teacher[cohort][0], device=DEVICE)
            graph = GraphDecode(model, past, lengths, initial, args.steps,
                                model.model.layers[0].mlp.scratch)
        for step in range(args.steps):
            ids = emitted.tolist() if teacher is None else teacher[cohort][step]
            if teacher is not None:
                emitted = torch.tensor(ids, device=DEVICE)
            while len(records) <= cohort:
                records.append([])
            records[cohort].append(ids)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            layers = []
            if args.profile_layers:
                for layer in model.model.layers:
                    layer.mlp.profile = layers
            pos = position_start + step
            if graph is not None:
                graph.set_input(emitted, pos, longest + step)
            torch.cuda.synchronize()
            start_wall = perf_counter()
            a.record()
            if graph is None:
                out = model(emitted[:, None], attention_mask=mask[:, :longest + step + 1],
                            position_ids=pos, past_key_values=past, use_cache=True, logits_to_keep=1)
                result = out.logits
                past = out.past_key_values
            else:
                result = graph.replay()
            b.record()
            b.synchronize()
            wall_ms = 1000 * (perf_counter() - start_wall)
            peak = torch.cuda.max_memory_allocated()
            mono_ms = sum(begin.elapsed_time(end) for begin, end in layers)
            if args.profile_layers:
                for layer in model.model.layers:
                    layer.mlp.profile = None
            if graph is not None and step == 0:
                rel = (result.float() - graph.expected.float()).norm() / graph.expected.float().norm()
                limit = .002
                value = rel.item()
                if value > limit:
                    worst = (result.float() - graph.expected.float()).abs().max().item()
                    expected_ids = graph.expected[:, -1].argmax(-1).tolist()
                    actual_ids = result[:, -1].argmax(-1).tolist()
                    raise RuntimeError(f'graph replay deviates on cohort {cohort}: '
                                       f'relative L2 {value}, max absolute {worst}, '
                                       f'expected and actual top-1 {expected_ids}, {actual_ids}')
            logits = result[:, -1]
            if not logits.isfinite().all().item():
                raise RuntimeError(f'non-finite logits at cohort {cohort}, step {step}')
            emitted = logits.argmax(-1)
            outputs[cohort].append(emitted.tolist())
            print(f'2,{cohort},{step},{a.elapsed_time(b):.6f},{wall_ms:.6f},{peak},{mono_ms:.6f}', flush=True)
    if args.tokens:
        args.tokens.write_text(json.dumps(records) + '\n')
    if args.outputs:
        args.outputs.write_text(json.dumps(outputs) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('prompts', type=Path)
    p.add_argument('--tiled', type=Path, required=True)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--tokens', type=Path)
    teachers = p.add_mutually_exclusive_group()
    teachers.add_argument('--teacher', type=Path)
    teachers.add_argument('--teacher-gguf', type=Path)
    p.add_argument('--outputs', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--profile-layers', action='store_true')
    mode.add_argument('--cuda-graph', action='store_true')
    p.add_argument('--campaign-dir', type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.tiled.resolve()))
    from monomoe_h20 import specialize

    source = jit_moe._get_monomoe_csrc_dir()
    fused = import_module('flashinfer.fused_moe.monomoe')
    torch.backends.cuda.matmul.allow_tf32 = True

    def initialize(flavor):
        os.environ['QWEN_MONO_BOUNDARY'] = flavor
        specialize(source, args.tiled / f'upstream-monomoe-{flavor}-grid64', 64)
        jit_moe.gen_monomoe_module.cache_clear()
        jit_moe.load_monomoe_module.cache_clear()
        return jit_moe.load_monomoe_module()

    if not args.campaign_dir:
        initialize(os.environ['QWEN_MONO_BOUNDARY'])
        run(args, load_model(args.model, alloc_scratchpad(DEVICE)))
        return

    flavors = ('flags', 'barrier', 'split_cached')
    modules = {flavor: initialize(flavor) for flavor in flavors}
    scratch = {flavor: torch.zeros(module.monomoe_scratchpad_size(), dtype=torch.uint8, device=DEVICE)
               for flavor, module in modules.items()}
    model = load_model(args.model, scratch['flags'])
    orders = (flavors, flavors[::-1], ('barrier', 'flags', 'split_cached'))
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    for rnd, order in enumerate(orders):
        for flavor in order:
            print(f'round {rnd}, {flavor}', flush=True, file=sys.stderr)
            module = modules[flavor]
            fused._get_monomoe_module = lambda: module
            scratch[flavor].zero_()
            for layer in model.model.layers:
                layer.mlp.scratch = scratch[flavor]
            stem = args.campaign_dir / f'{rnd}-{flavor}'
            args.outputs = stem.with_suffix('.out.json')
            partial = stem.with_suffix('.csv.part')
            with partial.open('w') as out, redirect_stdout(out):
                run(args, model)
            partial.rename(stem.with_suffix('.csv'))


if __name__ == '__main__':
    main()
