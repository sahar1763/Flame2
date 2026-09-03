#!/usr/bin/env python3
"""
generate_fig11_tradeoff.py
==========================

Render Report Figure 11 (performance-latency trade-off) so it matches Table 11
exactly. The two axes deliberately come from DIFFERENT real sources, mirroring
the table:

  * y-axis  Fire F1-score on the SmokeFire_unified crop-pipeline set
            (TensorRT FP16, single sqrt(2) crop, 8,120 images) -- the grading
            benchmark. Read from results_master_long.csv, dataset_view =
            SmokeFire_unified (FireSmoke + FireSmokeNEW summed confusion matrix).
  * x-axis  Mean MODEL inference time over the IAI dataset (TensorRT FP16,
            single crop, 1920x1080 operational frames). Read from
            crop_results/IAI_Datasets/<model>_tensorrt_crops1/summary.csv,
            column mean_model_inference_ms (forward pass only, matches Table 10/11).

Marker area is proportional to the FP16 engine size (trt_fp16_mb from
model_complexity_table.csv). Nothing is hard-coded; every number is read from a
real result file.

The output goes to figures_combined_iai_firesmoke/ (a dedicated folder) so it is
never confused with the pure-IAI plots in figures_iai_filtered/, which use IAI
performance on BOTH axes.

Usage:
  python generate_fig11_tradeoff.py
  python generate_fig11_tradeoff.py --out figures_combined_iai_firesmoke/fig11_tradeoff_tensorrt.png
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["mobilenet_v2", "resnet18", "alexnet", "resnet34", "resnet50", "vgg16"]


def load_f1_smokefire(master_csv):
    """SmokeFire_unified fire F1 per model (TensorRT FP16, single crop)."""
    m = pd.read_csv(master_csv)
    sf = m[(m.dataset_view == "SmokeFire_unified") & (m.backend == "tensorrt")
           & (m.crop_mode == "crops") & (m.num_crops == 1)]
    return dict(zip(sf.model, sf.fire_f1))


def load_latency_iai(results_dir):
    """IAI mean MODEL inference time per model (TensorRT FP16, single crop)."""
    lat = {}
    for mdl in MODELS:
        path = os.path.join(results_dir, f"{mdl}_tensorrt_crops1", "summary.csv")
        if os.path.isfile(path):
            lat[mdl] = float(pd.read_csv(path).iloc[0]["mean_model_inference_ms"])
    return lat


def load_engine_mb(complexity_csv):
    """FP16 engine size (MB) per model."""
    mc = pd.read_csv(complexity_csv)
    return dict(zip(mc.model, mc.trt_fp16_mb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="results_master_long.csv",
                    help="results_master_long.csv (SmokeFire_unified F1 source)")
    ap.add_argument("--iai_dir", default="crop_results/IAI_Datasets",
                    help="dir with <model>_tensorrt_crops1/summary.csv (IAI latency)")
    ap.add_argument("--complexity", default="figures_iai_filtered/model_complexity_table.csv",
                    help="model_complexity_table.csv (FP16 engine sizes)")
    ap.add_argument("--out", default="figures_combined_iai_firesmoke/fig11_tradeoff_tensorrt.png",
                    help="output PNG path")
    args = ap.parse_args()

    f1 = load_f1_smokefire(args.master)
    lat = load_latency_iai(args.iai_dir)
    eng = load_engine_mb(args.complexity)

    models = [m for m in MODELS if m in f1 and m in lat and m in eng]
    if len(models) < 2:
        raise SystemExit("Not enough models with all three sources available.")

    xs = [lat[m] for m in models]
    ys = [f1[m] for m in models]
    es = [eng[m] for m in models]

    # Same styling as generate_figures.generate_tradeoff_scatter (Fig 11):
    # bubble area scaled to the plotted quantity, uniform steelblue markers,
    # no per-point highlighting.
    emax = max(es)
    sizes = [e / emax * 400 + 80 for e in es]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=sizes, c="steelblue", edgecolors="black",
               alpha=0.7, zorder=5)

    for m, x, y, e in zip(models, xs, ys, es):
        ax.annotate(f"{m}\n({e:.1f} MB)", (x, y),
                    textcoords="offset points", xytext=(10, -5), fontsize=9)

    ax.set_xlabel("Mean Inference Time [ms]", fontsize=12)
    ax.set_ylabel("Fire F1-Score [%]", fontsize=12)
    ax.set_title("Accuracy–Latency Trade-Off", fontsize=13)
    ax.grid(True, alpha=0.3)

    ymin, ymax = min(ys), max(ys)
    ymar = (ymax - ymin) * 0.1 if ymax > ymin else 1
    ax.set_ylim(ymin - ymar, ymax + ymar)
    ax.set_xlim(0, max(xs) * 1.15)

    ax.text(0.98, 0.02, "Bubble size ∝ FP16 engine size",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            fontstyle="italic", color="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")
    print("\nData plotted (matches Table 11):")
    print(f"  {'model':14s}{'F1(SmokeFire)':>15s}{'infer_IAI[ms]':>15s}{'FP16[MB]':>12s}")
    for m in sorted(models, key=lambda k: -f1[k]):
        print(f"  {m:14s}{f1[m]:15.2f}{lat[m]:15.2f}{eng[m]:12.2f}")


if __name__ == "__main__":
    main()
