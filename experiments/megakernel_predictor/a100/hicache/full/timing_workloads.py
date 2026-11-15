"""Freeze a new set of complete Qwen3 prefixes and append requests."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from requests_data import TOPICS


SHAPES = ((4096, 512), (8192, 128), (8192, 512), (8192, 1024),
          (8192, 2048), (12288, 512), (12288, 1024))


def make_requests(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    results = {"calibration": [], "heldout": []}
    for phase, count in (("calibration", 1), ("heldout", 3)):
        for prefix, suffix in SHAPES:
            for i in range(count):
                label = f"timing-{phase}-{prefix}-{suffix}-{i}"
                rng = random.Random(label)
                text = f"Notebook entry {label}. " + " ".join(rng.choices(TOPICS, k=850))
                tokens = tokenizer.encode(text, add_special_tokens=False)
                followup = tokenizer.encode(f"\nSummarize the experiments in {label}.\n",
                                            add_special_tokens=False)
                assert len(tokens) >= prefix + suffix
                ids = tokens[:prefix + suffix - len(followup)] + followup
                digest = sha256(json.dumps(ids).encode()).hexdigest()
                results[phase].append(dict(label=label, prefix=prefix, suffix=suffix,
                                           ids=ids, sha256=digest))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(make_requests(args.model), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
