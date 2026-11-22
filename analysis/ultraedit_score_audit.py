"""Sample UltraEdit Parquet score columns without downloading the image columns.

Only selected image pairs are fetched through the dataset viewer for contact
sheets. Requires DuckDB, requests, numpy, matplotlib, and Pillow locally.
"""

import argparse
import csv
import io
import json
import random
import re
import time
import textwrap
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


VIEWER_URL = "https://datasets-server.huggingface.co/rows"
PARQUET_BASE = "https://huggingface.co/api/datasets/BleachNick/UltraEdit_500k/parquet/default/FreeForm"
DATASET = "BleachNick/UltraEdit_500k"
SCORES = (
    "clip_sim_image", "clip_sim_source", "clip_sim_target", "clip_sim_dir",
    "dinov2_sim", "ssim",
)
ADD_PATTERN = re.compile(r"\b(add|insert|include|introduce|place|put|attach)\b")
REPLACE_PATTERN = re.compile(r"\b(replace|swap|substitute|exchange)\b")
COLOR_PATTERN = re.compile(r"\b(color|colour|hue|tint|shade)\b")


def is_training_color(prompt):
    """Match the training loader's precedence for its Change Color category."""
    text = prompt.casefold()
    return not (ADD_PATTERN.search(text) or REPLACE_PATTERN.search(text)) and bool(
        COLOR_PATTERN.search(text)
    )


def fetch_shard(connection, shard):
    url = f"{PARQUET_BASE}/{shard}.parquet"
    columns = "file_row_number, idx, edit_prompt, source_caption, target_caption, " + ", ".join(SCORES)
    records = connection.execute(
        f"SELECT {columns} FROM read_parquet(?, file_row_number=true)", [url]
    ).fetchall()
    names = [item[0] for item in connection.description]
    shard_start = shard * 1282 + min(shard, 20)
    return [dict(zip(names, values), row_idx=shard_start + values[0]) for values in records]


def fetch_image_urls(row):
    params = {
        "dataset": DATASET, "config": "default", "split": "FreeForm",
        "offset": row["row_idx"], "length": 1,
    }
    for attempt in range(6):
        try:
            response = requests.get(VIEWER_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()["rows"][0]["row"]
            if payload["idx"] != row["idx"]:
                raise RuntimeError(f"Row mismatch at {row['row_idx']}")
            row["source_image"] = payload["source_image"]
            row["edited_image"] = payload["edited_image"]
            return
        except (requests.RequestException, ValueError, KeyError, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(3 * 2 ** attempt)


def summarize(rows):
    result = {}
    for score in SCORES:
        values = np.array([row[score] for row in rows], dtype=np.float64)
        result[score] = {
            "n": int(np.isfinite(values).sum()),
            "min": float(np.nanmin(values)),
            "p05": float(np.nanquantile(values, 0.05)),
            "p10": float(np.nanquantile(values, 0.10)),
            "median": float(np.nanmedian(values)),
            "mean": float(np.nanmean(values)),
            "p90": float(np.nanquantile(values, 0.90)),
            "p95": float(np.nanquantile(values, 0.95)),
            "max": float(np.nanmax(values)),
        }
    for label, predicate in {
        "ip2p_image_ge_075": lambda x: x["clip_sim_image"] >= 0.75,
        "ip2p_captions_ge_020": lambda x: min(x["clip_sim_source"], x["clip_sim_target"]) >= 0.2,
        "ip2p_direction_ge_020": lambda x: x["clip_sim_dir"] >= 0.2,
        "ip2p_all_three": lambda x: x["clip_sim_image"] >= 0.75
                                  and min(x["clip_sim_source"], x["clip_sim_target"]) >= 0.2
                                  and x["clip_sim_dir"] >= 0.2,
        "strict_image_ge_090": lambda x: x["clip_sim_image"] >= 0.9,
        "strict_dino_ge_090": lambda x: x["dinov2_sim"] >= 0.9,
        "strict_ssim_ge_080": lambda x: x["ssim"] >= 0.8,
    }.items():
        result[label] = sum(bool(predicate(row)) for row in rows)
    return result


def plot_distributions(all_rows, color_rows, path):
    metrics = ["clip_sim_image", "clip_sim_source", "clip_sim_target", "clip_sim_dir"]
    thresholds = [0.75, 0.2, 0.2, 0.2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, threshold in zip(axes.flat, metrics, thresholds):
        full = np.array([row[metric] for row in all_rows], dtype=float)
        color = np.array([row[metric] for row in color_rows], dtype=float)
        edges = np.linspace(min(full.min(), color.min()), max(full.max(), color.max()), 26)
        ax.hist(full, bins=edges, density=True, alpha=0.45, label=f"All ({len(all_rows):,})")
        ax.hist(color, bins=edges, density=True, alpha=0.55,
                label=f"Color ({len(color_rows):,})")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"IP2P-like cutoff {threshold:g}")
        ax.set_title(metric)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("UltraEdit score distributions: stratified shard sample")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def download_image(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as image:
        return image.convert("RGB").copy()


def contact_sheet(selected, metric, path):
    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 3.1 * len(selected)),
                             gridspec_kw={"width_ratios": [2.2, 2, 2]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.02,
                        wspace=0.18, hspace=0.26)
    for index, (group, row) in enumerate(selected):
        source = download_image(row["source_image"]["src"])
        target = download_image(row["edited_image"]["src"])
        label_ax = axes[index, 0]
        label_ax.axis("off")
        label = (f"{group.upper()} {metric}\n{row[metric]:.3f}\n"
                 f"row {row['row_idx']}\n\n"
                 + textwrap.fill(row["edit_prompt"].replace("\n", " "), width=28)
                 + f"\n\nDINO {row['dinov2_sim']:.3f} · SSIM {row['ssim']:.3f}")
        label_ax.text(0.03, 0.95, label, transform=label_ax.transAxes,
                      ha="left", va="top", fontsize=9)
        for ax, image, title in zip(axes[index, 1:], (source, target), ("Source", "Target")):
            ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=24,
                        help="Number of stratified Parquet shards to inspect")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rerender", action="store_true",
                        help="Recreate contact sheets from saved selected-row metadata")
    args = parser.parse_args()
    if not 1 <= args.shards <= 390:
        parser.error("--shards must be between 1 and 390")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.rerender:
        with (args.out / "sample_scores.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        all_rows = [{score: float(row[score]) for score in SCORES} for row in rows]
        color_rows = [row for row, original in zip(all_rows, rows)
                      if original["is_training_color"] == "True"]
        plot_distributions(all_rows, color_rows, args.out / "score_distributions.png")
        for metric in ("clip_sim_image", "clip_sim_dir"):
            selected_rows = json.loads((args.out / f"examples_{metric}.json").read_text())
            selected = [("low" if index < 3 else "high", row)
                        for index, row in enumerate(selected_rows)]
            for _, row in selected:
                fetch_image_urls(row)
            contact_sheet(selected, metric, args.out / f"examples_{metric}.png")
        return

    rng = random.Random(42)
    shards = [rng.randrange(390 * i // args.shards,
                            390 * (i + 1) // args.shards)
              for i in range(args.shards)]
    connection = duckdb.connect()
    connection.execute("PRAGMA disable_progress_bar")
    rows = []
    for index, shard in enumerate(shards, start=1):
        rows.extend(fetch_shard(connection, shard))
        print(f"Scanned {index}/{len(shards)} metadata shards ({len(rows):,} rows)", flush=True)
    color_rows = [row for row in rows if is_training_color(row["edit_prompt"])]
    if len(color_rows) < 20:
        raise RuntimeError("Too few color examples to characterize score tails")

    with (args.out / "sample_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_idx", "idx", "is_training_color",
                                                    *SCORES, "edit_prompt", "source_caption",
                                                    "target_caption"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (is_training_color(row["edit_prompt"])
                                   if key == "is_training_color" else row.get(key))
                             for key in writer.fieldnames})

    summary = {
        "dataset": DATASET,
        "sampling": f"{len(shards)} stratified Parquet shards across 390; score and text columns only",
        "shards": shards,
        "sampled_rows": len(rows),
        "sampled_color_rows": len(color_rows),
        "all": summarize(rows),
        "color": summarize(color_rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_distributions(rows, color_rows, args.out / "score_distributions.png")

    for metric in ("clip_sim_image", "clip_sim_dir"):
        ordered = sorted(color_rows, key=lambda row: row[metric])
        selected = [("low", row) for row in ordered[:3]] + [
            ("high", row) for row in ordered[-3:][::-1]
        ]
        for _, row in selected:
            fetch_image_urls(row)
        contact_sheet(selected, metric, args.out / f"examples_{metric}.png")
        (args.out / f"examples_{metric}.json").write_text(
            json.dumps([{key: row[key] for key in ("row_idx", "idx", "edit_prompt", *SCORES)}
                        for _, row in selected], indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
