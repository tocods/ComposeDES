"""Deterministic token IDs shared by all full-model benchmark paths."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


SHAPES = ((4096, 128), (4096, 512), (8192, 128), (8192, 512), (8192, 2048))
TOPICS = (
    "An engineer inspects a suspension bridge at dawn. The team records the "
    "temperature and the movements of each joint as traffic begins.",
    "A librarian catalogs maps collected during a long expedition. Her notes "
    "describe where the team crossed rivers and how their route changed.",
    "An orchestra rehearses a new symphony. The players record the tempo and "
    "dynamics of each passage before trying an alternative interpretation.",
    "A botanist surveys vegetation along a railway. He compares distributions "
    "of plants in sunlit locations with samples from the shaded embankment.",
)


def generate(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    data = {}
    for phase, count in (("calibration", 1), ("heldout", 2)):
        data[phase] = []
        for prefix, suffix in SHAPES:
            for i in range(count):
                label = f"{phase}-{prefix}-{suffix}-{i}"
                text = f"Archive record {label}. " + " ".join(
                    TOPICS[(j + i + int(phase == "heldout")) % len(TOPICS)] for j in range(640))
                source = tokenizer.encode(text, add_special_tokens=False)
                followup = tokenizer.encode(f"\nSummarize the main observations for {label}.\n",
                                            add_special_tokens=False)
                assert len(source) >= prefix + suffix
                data[phase].append(dict(label=label, prefix=prefix, suffix=suffix,
                                        ids=source[:prefix + suffix - len(followup)] + followup))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.model), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
