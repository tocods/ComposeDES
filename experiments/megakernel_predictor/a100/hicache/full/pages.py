"""Python bindings for the common FA2 page tiles and MPK readiness scheduler."""
from ctypes import CDLL, c_int, c_void_p
from pathlib import Path

import torch
from flashinfer.cascade import merge_states
from flashinfer.prefill import single_prefill_with_kv_cache as prefill


def check(value):
    if value:
        raise RuntimeError(f"MPK page runtime returned CUDA status {value}")


class PageKernel:
    def __init__(self, q, prefix, resident, group):
        library = CDLL(str(Path(__file__).with_name("libmpk_pages.so")))
        ptr = c_void_p
        abi = {
            "pages_init": [c_int] * 4,
            "pages_mpk": [ptr] * 7,
            "pages_dual": [ptr] * 5 + [c_int, ptr],
            "pages_publish": [ptr, c_int, ptr],
            "pages_wait": [],
            "pages_finalize": [],
        }
        for name, arguments in abi.items():
            method = getattr(library, name)
            method.argtypes = arguments
            method.restype = None if name == "pages_finalize" else c_int
        count = library.pages_init(q, prefix, resident, group)
        if count < 1:
            raise RuntimeError(f"Cannot initialize MPK page runtime: {count}")
        self.library = library
        self.count = count
        self.early = 1 + int(resident != 0)
        self.output = torch.empty((count, q, 32, 128), dtype=torch.bfloat16, device="cuda")
        self.lse = torch.empty((count, q, 32), dtype=torch.float32, device="cuda")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.library.pages_finalize()

    def publish(self, flag, index):
        stream = torch.cuda.current_stream().cuda_stream
        check(self.library.pages_publish(flag.data_ptr(), index, stream))

    def part(self, q, k, v, index):
        stream = torch.cuda.current_stream().cuda_stream
        check(self.library.pages_dual(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                      self.output.data_ptr(), self.lse.data_ptr(), index, stream))

    def merge(self, method="flashinfer"):
        if method == "direct":
            from merge_pages import merge_part_major
            return merge_part_major(self.output, self.lse)
        if method != "flashinfer":
            raise ValueError(f"Unknown attention merge method: {method}")
        return merge_states(self.output.transpose(0, 1).contiguous(),
                            self.lse.transpose(0, 1).contiguous())[0]

    def attention(self, q, k, v, flag, events, candidate, *, wait_for_workers=True,
                  merge_method="flashinfer"):
        stream = torch.cuda.current_stream()
        if candidate == "mpk":
            check(self.library.pages_mpk(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                        self.output.data_ptr(), self.lse.data_ptr(),
                                        flag.data_ptr(), stream.cuda_stream))
            # The launcher already orders both persistent streams before the
            # merge. The default also retires the region on the CPU.
            if wait_for_workers:
                check(self.library.pages_wait())
        else:
            for i in range(self.count):
                if i >= self.early:
                    stream.wait_event(events[i - self.early])
                self.part(q, k, v, i)
        return self.merge(merge_method)


@torch.inference_mode()
def smoke():
    stream = torch.cuda.Stream()
    copy = torch.cuda.Stream()
    prefix, resident, group, rows = 8192, 4096, 1024, 129
    with torch.cuda.stream(stream), PageKernel(rows, prefix, resident, group) as blocks:
        q = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
        flag = torch.zeros(1, device="cuda", dtype=torch.int32)
        host = [torch.randn((prefix + rows, 8, 128), dtype=torch.bfloat16).pin_memory()
                for _ in range(2)]
        k, v = (x.cuda() for x in host)
        reference = prefill(q, k, v, causal=True, backend="fa2")
        for candidate in ("dual", "mpk"):
            flag.zero_()
            k[resident:prefix].fill_(float("nan"))
            v[resident:prefix].fill_(float("nan"))
            copy.wait_stream(stream)
            events = []
            with torch.cuda.stream(copy):
                for group_id, start in enumerate(range(resident, prefix, group), 1):
                    end = min(start + group, prefix)
                    for dst, src in zip((k, v), host):
                        dst[start:end].copy_(src[start:end], non_blocking=True)
                    if candidate == "mpk":
                        blocks.publish(flag, group_id)
                    else:
                        event = torch.cuda.Event()
                        event.record()
                        events.append(event)
            print(f"Running {candidate}", flush=True)
            result = blocks.attention(q, k, v, flag, events, candidate)
            abs_error = (reference.float() - result.float()).abs().max().item()
            print(candidate, abs_error, flush=True)
            assert torch.allclose(result, reference, atol=0.003, rtol=0.02)


if __name__ == "__main__":
    smoke()
