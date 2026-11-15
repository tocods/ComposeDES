"""Calibrate and evaluate mixed-residency Qwen3 prefix recovery."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter_ns

import torch
from transformers import AutoTokenizer

from requests_data import TOPICS
from run_qwen import Prefill, micros
from selector import select


SHAPES = ((8192, 128), (8192, 512), (12288, 512))
GROUPS = (512, 1024, 2048)


def case_key(row):
    return "/".join(str(row[k]) for k in ("prefix", "suffix", "resident", "group"))


def requests(model, validation=False):
    tokenizer = AutoTokenizer.from_pretrained(model)
    splits = (("validation", 3),) if validation else (("calibration", 1), ("heldout", 2))
    groups = (2048,) if validation else GROUPS
    cohort = {phase: [] for phase, _ in splits}
    for phase, prompts in splits:
        for prefix, suffix in SHAPES:
            for sample in range(prompts):
                label = f"granularity-{phase}-{prefix}-{suffix}-{sample}"
                rng = random.Random(label)
                text = f"Measurement notebook {label}. " + " ".join(rng.choices(TOPICS, k=950))
                ids = tokenizer.encode(text, add_special_tokens=False)
                question = tokenizer.encode("\nSummarize the experimental observations.\n",
                                            add_special_tokens=False)
                assert len(ids) >= prefix + suffix
                ids = ids[:prefix + suffix - len(question)] + question
                for resident in (0, prefix // 2):
                    for group in groups:
                        cohort[phase].append(dict(label=f"{label}-{resident}-{group}", ids=ids,
                                                 prefix=prefix, suffix=suffix,
                                                 resident=resident, group=group))
    return cohort


def measure(model, row, frozen, repeats):
    profile = frozen["profile"]
    n = model.n
    whole = [False] * n
    first = [True] + [False] * (n - 1)
    split = [True] * n
    dual_mask = frozen["masks"]["dual"]
    mpk_mask = frozen["masks"]["mpk"]
    candidates = {
        "bulk": ("coalesced", whole, "direct"),
        "bulk_grouped": ("dual", whole, "direct"),
        "dual_first": ("dual", first, "direct"),
        "dual_all": ("dual", split, "direct"),
        "dual_legacy_all": ("dual", split, "flashinfer"),
        "dual_select": ("dual", dual_mask, "direct"),
        "mpk_select": ("mpk", mpk_mask, "direct"),
        "mpk_matched": ("mpk", dual_mask, "direct"),
    }

    approximate = frozen.get("trim_us")
    if approximate is not None:
        candidates.update(dual_trim=("dual", frozen["trim_mask"], "direct"),
                          dual_cached=("dual", dual_mask, "direct"))

    def run(backend, mask, merge_method, grouped=False):
        return model.run(backend, mask, merge_method=merge_method,
                         coalesce_whole=not grouped,
                         defer_layers=range(n) if backend == "mpk" else ())

    reference = run(*candidates["bulk"])
    ref_token = reference.argmax().item()
    reference_top = reference.float().flatten().topk(2)
    margin = (reference_top.values[0] - reference_top.values[1]).item()
    checked = {}
    for name, config in candidates.items():
        for k, v in model.device:
            k[model.resident:model.prefix].fill_(float("nan"))
            v[model.resident:model.prefix].fill_(float("nan"))
        logits = run(*config, grouped=name == "bulk_grouped")
        assert logits.isfinite().all().item(), name
        error = (logits.float() - reference.float()).abs().max().item()
        assert error < 1.0, (name, error)
        token = logits.argmax().item()
        checked[name] = (error, token, reference.float().flatten()[token].item())

    for repetition in range(repeats):
        ordered = list(candidates)
        random.Random(f'{row["label"]}-{repetition}').shuffle(ordered)
        for name in ordered:
            backend, mask, merge_method = candidates[name]
            model.compute.synchronize()
            start = perf_counter_ns()
            choice = 0.
            if name in ("dual_select", "mpk_select", "mpk_matched", "dual_trim"):
                planner = "mpk" if name == "mpk_select" else "dual"
                selected, _, _ = select(profile, planner,
                                        trim_us=approximate if name == "dual_trim" else 0.)
                choice = micros(start)
                assert selected == mask
            token = run(backend, mask, merge_method, grouped=name == "bulk_grouped").argmax().item()
            duration = micros(start)
            error, checked_token, ref_logit_of_choice = checked[name]
            yield dict(label=row["label"], case=case_key(row), candidate=name, repeat=repetition,
                       first_token=token, reference_token=ref_token, logit_error=error,
                       reference_margin=margin, checked_token=checked_token,
                       reference_choice_gap=reference_top.values[0].item() - ref_logit_of_choice,
                       readiness_layers=sum(mask), ttft_us=duration, selector_us=choice)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--mode", choices=("generate", "calibration", "contention", "heldout"), required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "generate":
        args.requests.write_text(json.dumps(requests(args.model, args.validation), separators=(",", ":")) + "\n")
        return
    data = json.loads(args.requests.read_text())
    if args.prediction:
        frozen = json.loads(args.prediction.read_text())
        assert frozen["requests_sha256"] == sha256(args.requests.read_bytes()).hexdigest()
    if args.mode == "heldout" and not args.prediction:
        parser.error("--prediction is required for heldout")
    model = Prefill(args.model)
    split = "calibration" if args.mode == "contention" else args.mode
    cohort = data["validation" if args.validation else split]
    with torch.cuda.stream(model.compute), args.output.open("w", buffering=1) as out:
        for row in cohort:
            print("Starting", row["label"], flush=True)
            with model.request(row):
                if args.mode == "calibration":
                    print(json.dumps(model.calibrate(all_profiles=True)), file=out)
                elif args.mode == "contention":
                    print(json.dumps(dict(case=case_key(row),
                                          timings=model.measure_contention())), file=out)
                else:
                    case = frozen["cases"][case_key(row)]
                    for result in measure(model, row, case, args.repetitions):
                        print(json.dumps(result), file=out)
            print("Finished", row["label"], flush=True)


if __name__ == "__main__":
    main()
