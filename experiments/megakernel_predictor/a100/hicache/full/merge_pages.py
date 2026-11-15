"""Merge base-2 attention states directly from the FA2 part-major layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def merge_kernel(values, lse, out, rows: tl.constexpr, parts: tl.constexpr,
                 width: tl.constexpr, part_block: tl.constexpr, row_block: tl.constexpr):
    row = tl.program_id(0) * row_block + tl.arange(0, row_block)
    part = tl.arange(0, part_block)
    dim = tl.arange(0, width)
    state = part[:, None] * rows + row[None, :]
    valid = (part[:, None] < parts) & (row[None, :] < rows)
    scores = tl.load(lse + state, valid, other=-float("inf"))
    max_score = tl.max(scores, axis=0)
    max_score = tl.where(max_score == -float("inf"), 0.0, max_score)
    weights = tl.exp2(scores - max_score[None, :])
    norm = tl.sum(weights, axis=0)
    weights = weights / tl.where(norm > 0, norm, 1.0)[None, :]
    vectors = tl.load(values + state[:, :, None] * width + dim[None, None, :],
                      valid[:, :, None], other=0).to(tl.float32)
    result = tl.sum(vectors * weights[:, :, None], axis=0)
    tl.store(out + row[:, None] * width + dim[None, :], result, row[:, None] < rows)


def merge_part_major(values, lse):
    parts, query, heads, width = values.shape
    output = torch.empty_like(values[0])
    row_block = 4
    rows = query * heads
    merge_kernel[(triton.cdiv(rows, row_block),)](
        values, lse, output, rows, parts, width, triton.next_power_of_2(parts),
        row_block, num_warps=4)
    return output
