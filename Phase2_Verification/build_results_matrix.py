#!/usr/bin/env python3
"""
build_results_matrix.py
=======================

Consolidate every RGB-verification configuration into a single decision table
so a model / backend / crop-strategy can be chosen at a glance.

For each of the 54 configurations (6 architectures x 3 backends x 3 crop
strategies) it reports TWO scored views side by side:

  * SmokeFire (unified)  -- FireSmokeDataset + FireSmokeNEWdataset combined by
                            SUMMING their confusion matrices (TP/FP/FN/TN) and
                            recomputing accuracy / precision / recall / F1 / FPR
                            on the pooled n = 6133 + 1987 = 8120 images. This is
                            the real held-out benchmark used to grade a model.
  * IAI (filtered)       -- the realized-scenario ILLUSTRATION set after removing
                            the 303 fires that no configuration could ever detect
                            (n = 2289). NOT a grading benchmark, shown only to
                            visualise behaviour on operational UAV frames.

Latency is the mean end-to-end pipeline time (mean_total_ms). For the unified
SmokeFire column it is the image-count weighted mean of the two datasets.

Inputs (real result files, nothing hard-coded):
  --combined   crop_results/combined_crop_summary.csv
               (contains FireSmokeDataset, FireSmokeNEWdataset, IAI_Datasets rows)
  --iai_filtered crop_results/IAI_Datasets_filtered/combined_crop_summary.csv
               (IAI after consensus-FN filtering, n = 2289)

Outputs:
  <out_csv>    wide machine-readable decision table (one row per config, the two
               scored views side by side; sort / filter in Excel or pandas)
  <out_txt>    human-readable fixed-width version of the decision table
  <out_master> COMPLETE long table: one row per (config x dataset_view) carrying
               EVERY metric (acc/prec/recall/F1/FPR/FNR + TP/FP/FN/TN) AND every
               pipeline-stage timing (mean/std/min/max for all 8 stages +
               total). dataset_view is one of:
                 FireSmoke, FireSmokeNEW, SmokeFire_unified,
                 IAI_original, IAI_filtered.
               This is the "pull anything into the report on demand" file.

Usage:
  python build_results_matrix.py
  python build_results_matrix.py --sort_by smokefire_f1
  python build_results_matrix.py --out_csv results_all_configs.csv \
                                 --out_txt results_all_configs.txt \
                                 --out_master results_master_long.csv
"""
import argparse
import csv
import os
from collections import defaultdict

METRIC_KEYS = ["accuracy", "fire_precision", "fire_recall", "fire_f1", "fpr"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def metrics_from_counts(tp, fp, fn, tn):
    """Recompute the standard metrics (percentages) from a confusion matrix."""
    n = tp + fp + fn + tn
    acc = 100.0 * (tp + tn) / n if n else 0.0
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    rec = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = 100.0 * fp / (fp + tn) if (fp + tn) else 0.0
    return dict(n=n, accuracy=acc, fire_precision=prec, fire_recall=rec,
                fire_f1=f1, fpr=fpr, TP=tp, FP=fp, FN=fn, TN=tn)


def read_summary(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def config_key(row):
    return (row["model"], row["backend"], row["crop_mode"], row["num_crops"])


# Pipeline stages in execution order (Stage 1..7 + total). Each has
# mean_/std_/min_/max_ columns in the summary CSVs.
STAGES = [
    "input_and_metadata", "region_validation", "crop_extraction",
    "resize_and_preprocess", "batch_construction", "model_inference",
    "majority_vote", "total",
]
STAGE_COLS = [f"{stat}_{s}_ms" for s in STAGES for stat in
              ("mean", "std", "min", "max")]
ID_COLS = ["model", "backend", "precision", "crop_mode", "num_crops",
           "crop_factors", "dataset_view"]
METRIC_COLS = ["num_images", "accuracy", "fire_precision", "fire_recall",
               "fire_f1", "fpr", "fnr", "TP", "FP", "FN", "TN"]
MASTER_COLS = ID_COLS + METRIC_COLS + STAGE_COLS


def master_row_from_csv(r, dataset_view):
    """Pass through a summary CSV row verbatim into master schema."""
    out = {
        "model": r["model"], "backend": r["backend"],
        "precision": r["precision"], "crop_mode": r["crop_mode"],
        "num_crops": r["num_crops"], "crop_factors": r["crop_factors"],
        "dataset_view": dataset_view,
        "num_images": r.get("num_images", ""),
        "accuracy": r.get("accuracy", ""),
        "fire_precision": r.get("fire_precision", ""),
        "fire_recall": r.get("fire_recall", ""),
        "fire_f1": r.get("fire_f1", ""),
        "fpr": r.get("fpr", ""), "fnr": r.get("fnr", ""),
        "TP": r.get("TP", ""), "FP": r.get("FP", ""),
        "FN": r.get("FN", ""), "TN": r.get("TN", ""),
    }
    for c in STAGE_COLS:
        out[c] = r.get(c, "")
    return out


def master_row_unified(k, counts, fire_rows, meta):
    """Build the SmokeFire_unified master row from the two FireSmoke rows.

    Metrics: recomputed from the SUMMED confusion matrix.
    Timing:  mean_/std_ = image-count weighted mean across the two datasets;
             min_ = min of mins; max_ = max of maxs (std weighting is an
             approximation, flagged in the docstring/header).
    """
    model, backend, crop_mode, num_crops = k
    m = metrics_from_counts(*counts)
    out = {
        "model": model, "backend": backend, "precision": meta["precision"],
        "crop_mode": crop_mode, "num_crops": num_crops,
        "crop_factors": meta["crop_factors"], "dataset_view": "SmokeFire_unified",
        "num_images": m["n"], "accuracy": f"{m['accuracy']:.2f}",
        "fire_precision": f"{m['fire_precision']:.2f}",
        "fire_recall": f"{m['fire_recall']:.2f}",
        "fire_f1": f"{m['fire_f1']:.2f}", "fpr": f"{m['fpr']:.2f}",
        "fnr": f"{100.0 * m['FN'] / (m['TP'] + m['FN']):.2f}"
        if (m["TP"] + m["FN"]) else "0.00",
        "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
    }
    ntot = sum(_i(r["num_images"]) for r in fire_rows) or 1
    for s in STAGES:
        wmean = sum(_f(r[f"mean_{s}_ms"]) * _i(r["num_images"])
                    for r in fire_rows) / ntot
        wstd = sum(_f(r[f"std_{s}_ms"]) * _i(r["num_images"])
                   for r in fire_rows) / ntot
        vmin = min(_f(r[f"min_{s}_ms"]) for r in fire_rows)
        vmax = max(_f(r[f"max_{s}_ms"]) for r in fire_rows)
        out[f"mean_{s}_ms"] = f"{wmean:.3f}"
        out[f"std_{s}_ms"] = f"{wstd:.3f}"
        out[f"min_{s}_ms"] = f"{vmin:.3f}"
        out[f"max_{s}_ms"] = f"{vmax:.3f}"
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Build a unified decision matrix (SmokeFire + IAI-filtered) "
                    "for every model/backend/crop configuration.")
    ap.add_argument("--combined",
                    default=os.path.join(here, "crop_results", "combined_crop_summary.csv"),
                    help="Combined summary with FireSmoke/FireSmokeNEW/IAI rows.")
    ap.add_argument("--iai_filtered",
                    default=os.path.join(here, "crop_results", "IAI_Datasets_filtered",
                                         "combined_crop_summary.csv"),
                    help="Filtered IAI combined summary (n=2289).")
    ap.add_argument("--out_csv",
                    default=os.path.join(here, "results_all_configs.csv"),
                    help="Wide machine-readable output CSV.")
    ap.add_argument("--out_txt",
                    default=os.path.join(here, "results_all_configs.txt"),
                    help="Human-readable fixed-width output table.")
    ap.add_argument("--out_master",
                    default=os.path.join(here, "results_master_long.csv"),
                    help="Complete long CSV: one row per (config x dataset_view) "
                         "with every metric and every timing stage.")
    ap.add_argument("--sort_by", default="smokefire_f1",
                    choices=["smokefire_f1", "smokefire_fpr", "smokefire_latency",
                             "iai_f1", "config"],
                    help="Row ordering for the text table (default smokefire_f1).")
    args = ap.parse_args()

    combined = read_summary(args.combined)
    iai_filt = read_summary(args.iai_filtered)

    # ---- unify FireSmoke + FireSmokeNEW by summing confusion matrices --------
    fire_counts = defaultdict(lambda: [0, 0, 0, 0])   # key -> [TP,FP,FN,TN]
    fire_lat = defaultdict(lambda: [0.0, 0])          # key -> [sum(lat*n), sum(n)]
    fire_rows = defaultdict(list)                     # key -> [FireSmoke rows]
    meta = {}
    for r in combined:
        folder = r.get("dataset_folder", "")
        if folder not in ("FireSmokeDataset", "FireSmokeNEWdataset"):
            continue
        k = config_key(r)
        c = fire_counts[k]
        c[0] += _i(r["TP"]); c[1] += _i(r["FP"])
        c[2] += _i(r["FN"]); c[3] += _i(r["TN"])
        n = _i(r["num_images"])
        fire_lat[k][0] += _f(r["mean_total_ms"]) * n
        fire_lat[k][1] += n
        fire_rows[k].append(r)
        meta[k] = dict(precision=r["precision"], num_crops=r["num_crops"],
                       crop_factors=r["crop_factors"])

    # ---- IAI filtered (already a single dataset per config) ------------------
    iai = {}
    for r in iai_filt:
        k = config_key(r)
        iai[k] = dict(m=metrics_from_counts(_i(r["TP"]), _i(r["FP"]),
                                            _i(r["FN"]), _i(r["TN"])),
                      lat=_f(r["mean_total_ms"]))

    # ---- assemble rows -------------------------------------------------------
    rows = []
    for k, c in fire_counts.items():
        model, backend, crop_mode, num_crops = k
        sf = metrics_from_counts(*c)
        sf_lat = fire_lat[k][0] / fire_lat[k][1] if fire_lat[k][1] else 0.0
        ia = iai.get(k, {}).get("m")
        ia_lat = iai.get(k, {}).get("lat", 0.0)
        rows.append(dict(
            model=model, backend=backend,
            precision=meta[k]["precision"], crop_mode=crop_mode,
            num_crops=num_crops, crop_factors=meta[k]["crop_factors"],
            sf=sf, sf_lat=sf_lat, ia=ia, ia_lat=ia_lat))

    def sort_val(row):
        if args.sort_by == "smokefire_f1":
            return -row["sf"]["fire_f1"]
        if args.sort_by == "smokefire_fpr":
            return row["sf"]["fpr"]
        if args.sort_by == "smokefire_latency":
            return row["sf_lat"]
        if args.sort_by == "iai_f1":
            return -(row["ia"]["fire_f1"] if row["ia"] else 0.0)
        return (row["model"], row["backend"], row["crop_mode"])
    rows.sort(key=sort_val)

    # ---- write wide CSV ------------------------------------------------------
    cols = ["model", "backend", "precision", "crop_mode", "num_crops",
            "crop_factors",
            "sf_n", "sf_accuracy", "sf_precision", "sf_recall", "sf_f1",
            "sf_fpr", "sf_TP", "sf_FP", "sf_FN", "sf_TN", "sf_latency_ms",
            "iai_n", "iai_accuracy", "iai_precision", "iai_recall", "iai_f1",
            "iai_fpr", "iai_TP", "iai_FP", "iai_FN", "iai_TN", "iai_latency_ms"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            sf, ia = r["sf"], r["ia"]
            w.writerow([
                r["model"], r["backend"], r["precision"], r["crop_mode"],
                r["num_crops"], r["crop_factors"],
                sf["n"], f"{sf['accuracy']:.2f}", f"{sf['fire_precision']:.2f}",
                f"{sf['fire_recall']:.2f}", f"{sf['fire_f1']:.2f}",
                f"{sf['fpr']:.2f}", sf["TP"], sf["FP"], sf["FN"], sf["TN"],
                f"{r['sf_lat']:.2f}",
                ia["n"] if ia else "", f"{ia['accuracy']:.2f}" if ia else "",
                f"{ia['fire_precision']:.2f}" if ia else "",
                f"{ia['fire_recall']:.2f}" if ia else "",
                f"{ia['fire_f1']:.2f}" if ia else "",
                f"{ia['fpr']:.2f}" if ia else "",
                ia["TP"] if ia else "", ia["FP"] if ia else "",
                ia["FN"] if ia else "", ia["TN"] if ia else "",
                f"{r['ia_lat']:.2f}" if ia else ""])

    # ---- write human-readable table -----------------------------------------
    hdr = ("{:<13} {:<9} {:<7} {:>3} | "
           "{:>6} {:>6} {:>6} {:>6} {:>6} | "
           "{:>6} {:>6} {:>6} {:>6} {:>6}")
    line = "-" * 120
    with open(args.out_txt, "w") as fh:
        fh.write("RGB VERIFICATION - ALL CONFIGURATIONS DECISION MATRIX\n")
        fh.write(line + "\n")
        fh.write("SmokeFire (unified) = FireSmoke + FireSmokeNEW pooled "
                 "confusion matrix, n=%d (grading benchmark)\n"
                 % (rows[0]["sf"]["n"] if rows else 0))
        fh.write("IAI (filtered)      = realized-scenario illustration, "
                 "n=%d (303 undetectable fires removed; NOT a benchmark)\n"
                 % (rows[0]["ia"]["n"] if rows and rows[0]["ia"] else 0))
        fh.write("Acc/Prec/Rec/F1/FPR in %%; Lat = mean end-to-end ms/frame. "
                 "Sorted by %s.\n" % args.sort_by)
        fh.write(line + "\n")
        fh.write("{:<13} {:<9} {:<7} {:>3} | {:^37} | {:^37}\n".format(
            "", "", "", "", "SmokeFire unified", "IAI filtered"))
        fh.write(hdr.format(
            "model", "backend", "crops", "n#",
            "SF_acc", "SF_prec", "SF_rec", "SF_f1", "SF_fpr",
            "IAI_acc", "IAI_prc", "IAI_rec", "IAI_f1", "IAI_fpr") + "\n")
        fh.write(line + "\n")
        for r in rows:
            sf, ia = r["sf"], r["ia"]
            crops = {"raw": "raw", "crops": "c%s" % r["num_crops"]}.get(
                r["crop_mode"], r["crop_mode"])
            fh.write(hdr.format(
                r["model"], r["backend"], crops, r["num_crops"],
                f"{sf['accuracy']:.1f}", f"{sf['fire_precision']:.1f}",
                f"{sf['fire_recall']:.1f}", f"{sf['fire_f1']:.1f}",
                f"{sf['fpr']:.1f}",
                f"{ia['accuracy']:.1f}" if ia else "-",
                f"{ia['fire_precision']:.1f}" if ia else "-",
                f"{ia['fire_recall']:.1f}" if ia else "-",
                f"{ia['fire_f1']:.1f}" if ia else "-",
                f"{ia['fpr']:.1f}" if ia else "-")
                + f"  | SF {r['sf_lat']:5.1f}ms  IAI {r['ia_lat']:5.1f}ms\n")
        fh.write(line + "\n")

    # ---- write COMPLETE master long table (everything) ----------------------
    # One row per (config x dataset_view). FireSmoke / FireSmokeNEW / IAI_original
    # are passed through verbatim from the combined summary; IAI_filtered from the
    # filtered summary; SmokeFire_unified is computed (summed CM + weighted time).
    view_map = {"FireSmokeDataset": "FireSmoke",
                "FireSmokeNEWdataset": "FireSmokeNEW",
                "IAI_Datasets": "IAI_original"}
    view_order = {"FireSmoke": 0, "FireSmokeNEW": 1, "SmokeFire_unified": 2,
                  "IAI_original": 3, "IAI_filtered": 4}
    master = []
    for r in combined:
        v = view_map.get(r.get("dataset_folder", ""))
        if v:
            master.append(master_row_from_csv(r, v))
    for r in iai_filt:
        master.append(master_row_from_csv(r, "IAI_filtered"))
    for k, counts in fire_counts.items():
        master.append(master_row_unified(k, counts, fire_rows[k], meta[k]))
    master.sort(key=lambda m: (m["model"], m["backend"], m["crop_mode"],
                               m["num_crops"], view_order.get(m["dataset_view"], 9)))
    with open(args.out_master, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MASTER_COLS)
        w.writeheader()
        w.writerows(master)

    print("Configurations: %d" % len(rows))
    print("Unified SmokeFire n = %d ; IAI filtered n = %d"
          % (rows[0]["sf"]["n"] if rows else 0,
             rows[0]["ia"]["n"] if rows and rows[0]["ia"] else 0))
    print("Master long rows: %d (%d configs x 5 dataset views)"
          % (len(master), len(rows)))
    print("Wrote %s" % args.out_csv)
    print("Wrote %s" % args.out_txt)
    print("Wrote %s" % args.out_master)


if __name__ == "__main__":
    main()
