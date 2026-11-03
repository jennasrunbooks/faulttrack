#!/usr/bin/env python3
"""Quick EDA on a few RCAEval cases per fault type (RE1 or RE2, metrics only).

    python eda_rcaeval.py data                          # all suites found
    python eda_rcaeval.py data --systems re1ob re2ob    # pick suites
    python eda_rcaeval.py data --per-fault 3 --out eda  # 3 cases per (system, fault)

Writes to --out:
  inventory.csv          one row per case: system, service, fault, rows, metrics, span, inject position, NaN %
  top_shifts.csv         top-10 robust z-shifts (post vs. pre injection) for each sampled case
  plots/<case>.png       the 4 most-shifted metrics around the injection time (dashed line)
and prints a summary: cases per system x fault, and which metric KIND shifts most per fault.

Reads local files only (no network). Loader logic matches spike_rcaeval.py (Zenodo metrics.json / data.csv,
or the HF parquet copy). Run inside a uv venv: uv run python eda_rcaeval.py data ...
"""
import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_rcaeval import (
    find_cases,
    inject_time,
    kind_of,
    load_metrics,
    read_metrics_json,
    robust_z,
)

warnings.filterwarnings("ignore")


def case_name(system, svc, fault, rep):
    return f"{system}_{svc}_{fault}_{rep}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--per-fault", type=int, default=2, help="cases per (system, fault) to inspect")
    ap.add_argument("--window", type=int, default=300, help="seconds before/after injection")
    ap.add_argument("--out", default="eda")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)
    idx = None
    cp = os.path.join(a.root, "cases.parquet")
    if os.path.exists(cp):
        c = pd.read_parquet(cp)
        tcol = next((x for x in c.columns if "inject" in x.lower()), None)
        if tcol is not None and "case" in c.columns:
            idx = dict(zip(c["case"], c[tcol], strict=False))

    inv, shifts, seen = [], [], {}
    for system, svc, fault, rep, d, p in sorted(find_cases(a.root)):
        if a.systems and system not in a.systems:
            continue
        try:
            df = load_metrics(p)
            t0 = inject_time(d, idx, os.path.basename(d))
        except (OSError, ValueError, KeyError) as e:   # unreadable or malformed case: skip it, keep going
            print("skip", d, e, file=sys.stderr)
            continue
        name = case_name(system, svc, fault, rep)
        if p.endswith(".json"):
            raw = read_metrics_json(p)
        else:
            raw = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        metric_cols = [c for c in raw.columns if not c.startswith("time")]
        inv.append({"case": name, "system": system, "service": svc, "fault": fault, "rows": len(df),
                        "metrics": df.shape[1] - 1, "span_s": int(df.time.max() - df.time.min()),
                        "inject_pos": round((t0 - df.time.min()) / max(1, df.time.max() - df.time.min()), 2),
                        "nan_pct": round(100 * raw[metric_cols].isna().mean().mean(), 2)})
        key = (system, fault)
        if seen.get(key, 0) >= a.per_fault:
            continue
        seen[key] = seen.get(key, 0) + 1

        pre = df[(df.time < t0) & (df.time >= t0 - a.window)].drop(columns="time")
        post = df[(df.time >= t0) & (df.time < t0 + a.window)].drop(columns="time")
        z = robust_z(pre, post)
        top = z.abs().sort_values(ascending=False).head(10).index
        for col in top:
            s_, k = kind_of(col)
            shifts.append({"case": name, "system": system, "injected_service": svc, "fault": fault, "metric": col,
                               "kind": k, "metric_service": s_, "z": round(float(z[col]), 1),
                               "pre_median": float(pre[col].median()), "post_median": float(post[col].median())})
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            w = df[(df.time >= t0 - a.window) & (df.time < t0 + a.window)]
            fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
            for ax, col in zip(axes, top[:4], strict=False):
                ax.plot(w.time - t0, w[col], lw=0.8)
                ax.axvline(0, ls="--", c="gray")
                ax.set_title(f"{col}  (z={z[col]:+.1f})", fontsize=9)
            axes[-1].set_xlabel("seconds from injection")
            fig.suptitle(f"{name}: injected {fault} on {svc}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "plots", f"{name}.png"), dpi=110)
            plt.close(fig)
        except ImportError:
            pass

    if not inv:
        sys.exit("no cases found — check the path/layout")
    inv = pd.DataFrame(inv)
    sh = pd.DataFrame(shifts)
    inv.to_csv(os.path.join(a.out, "inventory.csv"), index=False)
    sh.to_csv(os.path.join(a.out, "top_shifts.csv"), index=False)

    print("cases per system x fault:\n", inv.groupby(["system", "fault"]).case.count().unstack(fill_value=0), "\n")
    print("rows / metrics / NaN% by system:\n", inv.groupby("system")[["rows", "metrics", "nan_pct"]].median(), "\n")
    if not sh.empty:
        topk = sh.groupby("case").head(3)
        print("most-shifted metric KIND per fault (share of each case's top-3 shifts):")
        print(pd.crosstab(topk.fault, topk.kind.fillna("other"), normalize="index").round(2), "\n")
        first = sh.groupby("case").head(1)
        top_svc = first.metric_service.fillna("").str.replace("-", "")
        hit = (top_svc == first.injected_service.str.replace("-", "")).mean()
        print(f"top shift is on the injected service in {hit:.0%} of sampled cases")
    print(f"\nwrote {a.out}/inventory.csv, {a.out}/top_shifts.csv, {a.out}/plots/")


if __name__ == "__main__":
    main()
