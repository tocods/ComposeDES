#!/usr/bin/env python3
"""Instrument the production tiled projection loops in a standalone profiler."""
from pathlib import Path
import subprocess
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected one anchor {old!r}')
    return source.replace(old, new)


root = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
kernel = out / 'resident-moe-trace.cu'
subprocess.run([sys.executable, str(root / 'tiled/generate.py'), str(root / 'resident-moe.cu'), str(kernel)], check=True)
text = kernel.read_text()
text = replace_once(text, 'struct State {', '''struct Stamp { unsigned long long begin, end; unsigned int routes; };
__device__ __forceinline__ unsigned long long clock_ns() {
    unsigned long long out;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(out));
    return out;
}
struct State {''')
text = replace_once(text, '    float score_scale;', '    float score_scale;\n    Stamp *stamps = nullptr;')
text = replace_once(text, '        tile_project<true, batch>(s, group, offset, width, scratch);', '''
        unsigned long long begin = 0;
        if (s.stamps && threadIdx.x == 0) begin = clock_ns();
        tile_project<true, batch>(s, group, offset, width, scratch);''')
text = replace_once(text, '        if constexpr (publish) {', '''
        if (s.stamps && threadIdx.x == 0) s.stamps[task] = {begin, clock_ns(), unsigned(group.count)};
        if constexpr (publish) {''')
text = replace_once(text, '        tile_project<false, batch>(s, s.routes->groups[index], task % count * width, width, nullptr);', '''
        unsigned long long begin = 0;
        if (s.stamps && threadIdx.x == 0) begin = clock_ns();
        tile_project<false, batch>(s, s.routes->groups[index], task % count * width, width, nullptr);
        if (s.stamps) {
            __syncthreads();
            if (threadIdx.x == 0)
                s.stamps[slots * ff / 32 + task] = {begin, clock_ns(), unsigned(s.routes->groups[index].count)};
        }''')
kernel.write_text(text)
profile = (root / 'stage_profile.cu').read_text()
profile = replace_once(profile, '#include "resident-moe.cu"', f'#include "{kernel.name}"')
start = profile.index('template<int batch>\nvoid measure')
end = profile.index('} // namespace', start)
profile = profile[:start] + (here / 'trace.inc').read_text() + profile[end:]
profile = replace_once(profile,
    'batch,layer,cohort,step,unique,route_us,xq_us,first_us,hq_us,down_us,reduce_us,empty_us,coop_us,barrier_us',
    'batch,layer,cohort,step,producers,trial,overlap,stage,tile,routes,weight_bytes,start_ns,end_ns')
(out / 'tile_trace.cu').write_text(profile)
