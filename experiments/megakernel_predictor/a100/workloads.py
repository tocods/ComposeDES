"""Freeze exact token prompts before calibration and split by content."""
import argparse
import hashlib
import json
from pathlib import Path


OUTPUTS = 17
CONTEXTS = (32, 96, 256, 512)
TOPICS = {
    "calibration": (
        "A museum conservator records the temperature, lighting, and humidity of a gallery. "
        "Each painting is inspected before exhibition and then photographed for the archive.",
        "A coastal village uses tidal charts to coordinate deliveries. Residents keep records "
        "of wind and rainfall throughout the year to decide when to repair the harbor.",
        "An urban farm rotates crops between several raised beds. Soil samples and watering "
        "logs help its volunteers compare the yield of different planting dates.",
        "An orchestra plans a concert series. The conductor budgets rehearsal time according "
        "to the complexity of each piece and the availability of musicians.",
    ),
    "heldout": (
        "A librarian is building an index of letters from an expedition. The letters include "
        "sketches, weather observations, and notes about where the group camped.",
        "An engineer tests a pedestrian bridge under a calibrated load. Sensors on each "
        "span record how the bridge settles when the load moves across it.",
        "A school cafeteria tracks how much bread and fruit goes unused. The staff adjust "
        "their weekly orders after comparing weekday attendance with meal preferences.",
        "An observatory measures changes in starlight. Its team uses several filters to "
        "separate temperature effects from variations in the telescope.",
        "Two bakers compare sourdough starters across seasons. They record the rise time, "
        "dough temperature, and mass of discarded starter for several batches.",
        "Train operators plan maintenance across two cities. They compare inspection "
        "windows, parts deliveries, and the availability of alternative routes.",
        "A botanist studies the spread of plants along an old railway. She surveys "
        "shaded and open sections and records the slope and soil composition.",
        "A city library offers a short course about maps. Students trace historic rivers "
        "and measure how the boundaries of nearby neighborhoods have changed.",
    ),
}


def prepare(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    suffix = tokenizer.encode("\nSummarize the main observation in plain English.\n",
                              add_special_tokens=False)
    data = {}
    for cohort, topics in TOPICS.items():
        data[cohort] = []
        for index, topic in enumerate(topics):
            context = CONTEXTS[index % len(CONTEXTS)]
            stem = tokenizer.encode((" " + topic) * 32, add_special_tokens=False)
            tokens = stem[:context - len(suffix)] + suffix
            assert len(tokens) == context
            data[cohort].append({
                "request": f"{cohort}-{index + 1}",
                "tokens": tokens,
                "context": context,
                "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            })
    assert len({r["sha256"] for rows in data.values() for r in rows}) == 12
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(prepare(args.model), indent=2) + "\n")


if __name__ == "__main__":
    main()
