#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

MODELS = {
    "WL-DNN": ("wl_dnn", "wl_dnn"),
    "Strong DNN": ("strong_dnn", "process_nn"),
    "Irrigation-aware DNN": ("direct_irr", "process_nn"),
    "Process gate (V13)": ("v13_process_gate", "process_nn"),
    "Gradient-informed NN": ("gradient_informed", "process_nn"),
    "Temporal CNN": ("cnn", "process_nn"),
    "Multi-scale CNN": ("multiscale", "process_nn"),
    "Hybrid NN-CNN": ("hybrid", "process_nn"),
    "DSGR-Net (V16)": ("v16_process_summary", "process_nn"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, (prefix, key) in MODELS.items():
        values = []
        for seed in (42, 43, 44):
            document = json.loads((args.input / f"{prefix}_seed{seed}.json").read_text())
            values.append(document["models"][key]["test"])
        row = {"model": label}
        for metric in ("r2", "rmse", "mae"):
            samples = [float(item[metric]) for item in values]
            row[f"{metric}_mean"] = statistics.mean(samples)
            row[f"{metric}_sample_sd"] = statistics.stdev(samples)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
