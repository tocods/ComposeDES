#!/usr/bin/env python3
"""Generate selectable CTA geometries and an expert-local resident kernel."""
import re
from pathlib import Path
import sys


def main(source, output):
    text = source.read_text()
    text = text.replace('#include <cstring>', '#include <cstring>\n#include <cstdlib>')
    text = text.replace('threads = 256, warp = 32, warps = threads / warp', 'threads = 256, warp = 32')
    text = text.replace('int(blockIdx.x) * warps + threadIdx.x / warp', 'int(blockIdx.x) * (blockDim.x / warp) + threadIdx.x / warp')
    text = text.replace('gridDim.x * warps', 'gridDim.x * (blockDim.x / warp)')
    text = text.replace('const int w = threadIdx.x / warp, lane = threadIdx.x % warp;',
                        'const int lane = threadIdx.x % warp;')
    text = text.replace('if (w < s.batch) {',
                        'for (int w = threadIdx.x / warp; w < s.batch; w += blockDim.x / warp) {')
    where = text.index('enum Stage')
    text = text[:where] + Path(__file__).with_name('expert-local.inc').read_text() + '\n' + text[where:]
    text = text.replace('GATE_CUT, FOUR };', 'GATE_CUT, FOUR, EXPERT_LOCAL, EXPERT_BARRIER };')
    text = text.replace('int blocks, State &s) {\n    constexpr bool multi', 'int blocks, State &s, int nthreads) {\n    constexpr bool multi')
    text = text.replace('blocks, threads, args, 0, stream', 'blocks, nthreads, args, 0, stream')
    text = text.replace('<<<grid, threads, 0, stream>>>', '<<<grid, nthreads, 0, stream>>>')
    text = text.replace('void dispatch(cudaStream_t stream, int blocks, int mode, State &s) {', '''void dispatch(cudaStream_t stream, int blocks, int mode, State &s, int first, int second) {''')
    # Keep the isolated routing, quantization, and reduction kernels fixed.
    def geometry(call):
        mask = call[1]
        if mask in ('ROUTE', 'XQ', 'HQ', 'REDUCE'):
            return call[0].replace(', s);', ', s, threads);')
        phase = 'second' if mask in ('HQ | DOWN | REDUCE', 'DOWN') else 'first'
        return call[0].replace(', s);', f', s, {phase});')
    text = re.sub(r'launch<batch, ([^>]+)>\(stream, blocks, s\);', geometry, text)
    text = text.replace('        default:\n            GGML_ABORT("invalid resident MoE cut");', '''        case EXPERT_LOCAL:
        case EXPERT_BARRIER: {
            GGML_ASSERT(blocks >= batch * top);
            void *args[] = {&s};
            const size_t scratch = batch * (ff * sizeof(float) + ff / warp * sizeof(block_q8_1));
            const void *kernel = mode == EXPERT_BARRIER
                ? reinterpret_cast<const void *>(expert_local<batch, true>)
                : reinterpret_cast<const void *>(expert_local<batch, false>);
            CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, blocks, first, args, scratch, stream));
            break;
        }
        default:
            GGML_ABORT("invalid resident MoE cut");''')
    text = text.replace("    const auto stream = ctx.stream();", """    const auto stream = ctx.stream();
    const auto width = [](const char *key) {
        const char *value = std::getenv(key);
        const int count = value ? std::atoi(value) : 256;
        GGML_ASSERT(count == 128 || count == 256 || count == 512);
        return count;
    };
    static const int first = width("QWEN_CTA_FIRST"), second = width("QWEN_CTA_SECOND");""")
    text = text.replace('stream, blocks, mode, s);','stream, blocks, mode, s, first, second);')
    output.write_text(text)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
