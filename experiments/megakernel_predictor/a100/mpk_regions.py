"""Build separate MPK executables for contiguous regions of one Qwen3-8B decode."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

MIRAGE = Path(os.environ["MIRAGE_HOME"])
sys.path.insert(0, str(MIRAGE / "demo" / "qwen3"))
from demo import grid_for_rmsnorm_linear_layer
from models.modeling_qwen3 import Qwen3ForCausalLM
import mirage as mi

MAX_SEQ = 1024
PAGE = 256
PREFILL_CHUNK = 8
RUNTIME_WORKERS = 80
BLOCK = (128, 1, 1)
LAYERS = 36
OPERATORS = 2 * LAYERS + 2
PLANS = {
    "mpk_full": (0, OPERATORS),
    "mpk_attn18_mlp18": (0, 36, OPERATORS),
    "mpk_layer18_19": (0, 37, OPERATORS),
}


def grid(n):
    return (n, 1, 1)


def interleave(weights, groups):
    """Use the same row order as Graph::shuffle_tensors."""
    return torch.cat([weight.reshape(groups, -1, weight.shape[1])
                      for weight in weights], dim=1).flatten(0, 1)


class Model:
    def __init__(self, path):
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            self.net = Qwen3ForCausalLM.from_pretrained(
                path, 1, max_num_pages=MAX_SEQ // PAGE, page_size=PAGE,
            ).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = self.net.config
        assert len(self.net.model.layers) == LAYERS
        self.workers, self.schedulers = mi.get_configurations_from_gpu(0)
        self.kv_plan = self.net.model.kv_plan
        self.kv_plan.max_num_pages = MAX_SEQ // PAGE
        self.cos, self.sin = (x[0].contiguous() for x in
                              self.net.model.rotary_emb(torch.arange(MAX_SEQ, device="cuda")[None]))
        self.lm_head = torch.nn.functional.pad(
            self.net.lm_head.weight, (0, 0, 0, 153600 - self.config.vocab_size))
        self.qkv = []
        self.gate_up = []
        for layer in self.net.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            self.qkv.append(interleave(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                self.config.num_key_value_heads))
            groups = grid_for_rmsnorm_linear_layer(2 * self.config.intermediate_size) // 2
            self.gate_up.append(interleave([mlp.gate_proj.weight, mlp.up_proj.weight], groups))

    def runner(self, max_queries):
        return Runner(self, max_queries)


class Runner:
    def __init__(self, model, max_queries):
        self.model = model
        self.n = max_queries
        self.regions = {}
        self.meta = model.kv_plan.build_meta_tensors(
            max_num_batched_requests=1, max_seq_length=MAX_SEQ)
        self.meta.update({
            "step": torch.zeros(1, dtype=torch.int32, device="cuda"),
            "tokens": torch.zeros((1, MAX_SEQ), dtype=torch.int64, device="cuda"),
            "input_tokens": torch.zeros((self.n, 1), dtype=torch.int64, device="cuda"),
            "output_tokens": torch.zeros((self.n, 1), dtype=torch.int64, device="cuda"),
            "num_new_tokens": torch.ones(1, dtype=torch.int32, device="cuda"),
            "prompt_lengths": torch.zeros(1, dtype=torch.int32, device="cuda"),
            "qo_indptr_buffer": torch.zeros(2, dtype=torch.int32, device="cuda"),
        })
        self.meta["paged_kv_indptr_buffer_0"].zero_()
        self.meta["paged_kv_indices_buffer_0"].copy_(
            torch.arange(MAX_SEQ // PAGE, device="cuda", dtype=torch.int32))
        self.hidden = {
            name: torch.empty((self.n, model.config.hidden_size), device="cuda")
            for name in ("embed_hidden", "attn_hidden", "mlp_hidden")
        }

    def setup(self, pos, ids):
        count = ids.numel()
        assert 0 < count <= self.n and pos + count <= MAX_SEQ
        seq = pos + count
        self.meta["input_tokens"][:count, 0].copy_(ids)
        self.meta["step"].fill_(pos)
        self.meta["qo_indptr_buffer"][1:].fill_(count)
        self.meta["paged_kv_indptr_buffer_0"][1:].fill_((seq + PAGE - 1) // PAGE)
        self.meta["paged_kv_last_page_len_buffer_0"].fill_((seq - 1) % PAGE + 1)

    def run(self, regions):
        for region in regions:
            region()
            # The worker and scheduler streams own persistent queues. Retire
            # both before enqueuing the next resident region.
            region.wait()

    def plan(self, name):
        boundaries = PLANS[name]
        return tuple(self.regions[start, end] for start, end in zip(boundaries, boundaries[1:]))

    def build(self, region, output):
        start, end = region
        model, n = self.model, self.n
        cfg = model.config
        hidden = cfg.hidden_size
        k_heads = cfg.num_key_value_heads
        q_size = cfg.num_attention_heads * cfg.head_dim
        mpk = mi.PersistentKernel(
            mode="online_notoken", world_size=1, mpi_rank=0,
            num_workers=model.workers, num_local_schedulers=model.schedulers,
            num_remote_schedulers=0, max_seq_length=MAX_SEQ,
            max_num_batched_requests=1, max_num_batched_tokens=n,
            max_num_pages=MAX_SEQ // PAGE, kv_groups=model.kv_plan.group_specs(),
            meta_tensors=self.meta, use_cutlass_kernel=True,
            profiler_tensor=None, trace_name=None, spec_decode_config=None,
        )

        def attach(name, tensor):
            return mpk.attach_input(torch_tensor=tensor, name=name)

        def scratch(name, width, dtype=mi.bfloat16):
            return mpk.new_tensor(dims=(n, width), dtype=dtype, name=name)

        input_ids = attach("input_token", self.meta["input_tokens"])
        output_ids = attach("output_token", self.meta["output_tokens"])
        embed, attn_hidden, mlp_hidden = (attach(name, buf) for name, buf in self.hidden.items())
        cos = attach("cos_position_embedding", model.cos)
        sin = attach("sin_position_embedding", model.sin)
        norm = scratch("rmsnorm_out", hidden)
        attn_in = scratch("attn_in", q_size + 2 * k_heads * cfg.head_dim)
        attn_out = scratch("attn_out", q_size)
        mlp_mid = scratch("mlp_mid", 2 * cfg.intermediate_size)
        silu = scratch("silu_mul_out", cfg.intermediate_size)

        if start == 0:
            mpk.embed_layer(input_ids, attach("embed_tokens", model.net.model.embed_tokens.weight),
                            embed, grid(1), BLOCK, input_source=1)

        for i, layer in enumerate(model.net.model.layers):
            attn_op, mlp_op = 2 * i + 1, 2 * i + 2
            prefix = f"layer_{i}_"
            if start <= attn_op < end:
                residual = embed if i == 0 else mlp_hidden
                w_norm = attach(prefix + "input_norm", layer.input_layernorm.weight)
                w_qkv = attach(prefix + "qkv", model.qkv[i])
                mpk.rmsnorm_layer(residual, w_norm, norm, grid(n), BLOCK)
                mpk.linear_layer(norm, w_qkv, attn_in,
                                 grid(grid_for_rmsnorm_linear_layer(w_qkv.dim(0))), BLOCK)
                group, slot = model.kv_plan.layer_info(i)
                cache = model.net.model.kv_cache
                attn = layer.self_attn
                mpk.paged_attention_layer(
                    attn_in, attach(prefix + "k_cache", cache[0][slot]),
                    attach(prefix + "v_cache", cache[1][slot]),
                    attach(prefix + "q_norm", attn.q_norm.weight),
                    attach(prefix + "k_norm", attn.k_norm.weight),
                    cos, sin, attn_out, (1, k_heads, 1), BLOCK, group_id=group)
                mpk.linear_with_residual_layer(
                    input=attn_out, weight=attach(prefix + "o", attn.o_proj.weight),
                    residual=residual, output=attn_hidden, grid_dim=grid(hidden // 64),
                    block_dim=BLOCK)
            if start <= mlp_op < end:
                w_norm = attach(prefix + "post_norm", layer.post_attention_layernorm.weight)
                w_gate_up = attach(prefix + "gate_up", model.gate_up[i])
                tasks = grid_for_rmsnorm_linear_layer(w_gate_up.dim(0))
                mpk.rmsnorm_layer(attn_hidden, w_norm, norm, grid(n), BLOCK)
                mpk.linear_layer(norm, w_gate_up, mlp_mid, grid(tasks), BLOCK)
                mpk.silu_mul_layer(mlp_mid, silu, grid(tasks // 2), BLOCK)
                mpk.linear_with_residual_layer(
                    input=silu, weight=attach(prefix + "down", layer.mlp.down_proj.weight),
                    residual=attn_hidden, output=mlp_hidden, grid_dim=grid(hidden // 64),
                    block_dim=BLOCK)

        if end == OPERATORS:
            logits = scratch("argmax_in", 153600)
            partial = (scratch("argmax_part_value", model.workers),
                       scratch("argmax_part_index", model.workers, mi.int64))
            mpk.rmsnorm_layer(mlp_hidden, attach("model_norm", model.net.model.norm.weight),
                              norm, grid(n), BLOCK)
            mpk.linear_layer(norm, attach("lm_head", model.lm_head), logits,
                             grid(model.workers), BLOCK)
            mpk.argmax_partial_layer(logits, partial, grid(model.workers), BLOCK,
                                     vocab_size=cfg.vocab_size)
            mpk.argmax_reduce_layer(partial, output_ids, grid(1), BLOCK)

        folder = output / f"q{n}-{start}-{end}"
        assert model.schedulers <= RUNTIME_WORKERS <= model.workers
        if (folder / "ready").exists():
            mpk.load_mpk_kernel(str(folder), runtime_num_workers=RUNTIME_WORKERS)
        else:
            mpk.compile(output_dir=str(folder), runtime_num_workers=RUNTIME_WORKERS)
            (folder / "ready").touch()
        self.regions[region] = mpk

    def prefill(self, ids, plan):
        for pos in range(0, ids.numel(), self.n):
            self.setup(pos, ids[pos:pos + self.n])
            self.run(plan)
            # Complete a prefill iteration before reusing its MPK queue and
            # event state for a new chunk.
            torch.cuda.current_stream().synchronize()
        return self.meta["output_tokens"][(ids.numel() - 1) % self.n, 0].item()


def compile_runners(model, output):
    prefill = model.runner(PREFILL_CHUNK)
    prefill.build((0, OPERATORS), output)
    decode = model.runner(1)
    for boundaries in PLANS.values():
        for region in zip(boundaries, boundaries[1:]):
            decode.build(region, output)
    return prefill, decode


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    model = Model(args.model)
    prefill, decode = compile_runners(model, args.build)
    # Pass an explicit stream pointer to the generated CUDA module and Torch.
    torch.cuda.synchronize()
    torch.cuda.set_stream(torch.cuda.Stream())
    print("Running smoke prefill", flush=True)
    ids = model.tokenizer("Explain why the sky is blue in two sentences.",
                          return_tensors="pt").input_ids[0].cuda()
    first = prefill.prefill(ids, prefill.plan("mpk_full"))
    print("Smoke prefill completed", flush=True)
    generated = {}
    for name in PLANS:
        print(f"Checking {name}", flush=True)
        token = torch.tensor([first], dtype=torch.int64, device="cuda")
        result = []
        for pos in range(ids.numel(), ids.numel() + 8):
            decode.setup(pos, token)
            decode.run(decode.plan(name))
            token = decode.meta["output_tokens"][0]
            result.append(token.item())
        generated[name] = result
    print(json.dumps({"first": first, "generated": generated}))
    assert all(ids == generated["mpk_full"] for ids in generated.values()), generated


if __name__ == "__main__":
    main()
