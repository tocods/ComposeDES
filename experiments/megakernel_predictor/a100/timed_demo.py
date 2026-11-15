"""Time the upstream Qwen3 demo with the same prefill-to-completion scope."""
from pathlib import Path
import os
import sys
import tempfile


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'upstream anchor changed: {old!r}')
    return source.replace(old, new)


upstream = Path(os.environ['MIRAGE_HOME']) / 'demo/qwen3/demo.py'
source = upstream.read_text()
source = replace(source,
                 '        mpk.compile(output_dir=args.output_dir)',
                 '''        if os.getenv("MPK_A100_REUSE") == "1":
            mpk.load_mpk_kernel(output_dir=args.output_dir)
        else:
            mpk.compile(output_dir=args.output_dir)''')
source = replace(source, '''        decode_limit = prompt_len + output_len
        for cur_pos in range(prompt_len, decode_limit):''',
                 '''        decode_limit = prompt_len + output_len
        # Warm up the two eager operator shapes before the measured request.
        # The measured prefill overwrites the same logical KV prefix.
        for start, end in ((0, prompt_len), (prompt_len - 1, prompt_len)):
            step.fill_(end - 1)
            model.forward(
                input_ids=tokens[:, start:end],
                position_embeddings=tuple(p[:, start:end] for p in position_embeddings),
                step=step, stream=stream,
            )
        torch.cuda.synchronize()
        starter.record()
        for cur_pos in range(prompt_len, decode_limit):''')
source = replace(source, '''            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()
''', '')
for mode in ('torch', 'mpk'):
    anchor = f'                "mode": "{mode}",'
    source = replace(source, anchor, anchor + '''
                "total_time_ms": run_time,
                "ms_per_output_token_including_prefill": run_time / tokens_generated,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),''')
sys.path.insert(0, str(upstream.parent))
with tempfile.TemporaryDirectory(prefix='mpk-demo-') as folder:
    script = Path(folder) / 'demo.py'
    script.write_text(source)
    sys.argv[0] = str(upstream)
    g = {'__name__': '__main__', '__file__': str(upstream)}
    exec(compile(source, str(script), 'exec'), g)
