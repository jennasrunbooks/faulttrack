#!/usr/bin/env python3
"""Option A feasibility spike: can we classify the FAULT TYPE of an RCAEval case from metrics?

Run on your laptop or in a devcontainer, inside a uv venv (the data hosts are blocked in Claude's sandbox).

Data: the Zenodo record (CC BY 4.0, DOI 10.5281/zenodo.14590730), one zip per suite:
    https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1   (likewise RE2-SS, RE2-TT, RE1-OB, ...)
The RE2 zips include multi-GB logs/traces. Extract only the metrics + inject_time files, e.g.
    unzip -l RE2-OB.zip | head -40                                   # check the real layout first
    unzip -o RE2-OB.zip '*inject_time.txt' '*metrics.json' '*data.csv' -d data/

    uv run python spike_rcaeval.py data                         # all suites found
    uv run python spike_rcaeval.py data --systems re2ob         # one suite
    uv run python spike_rcaeval.py data --loso                  # also leave-one-system-out (main result)

Layouts understood: Zenodo <SUITE>/<service>_<fault>/<n>/{metrics.json|data.csv|metrics.csv} + inject_time.txt,
and the Hugging Face parquet copy (re2ob_<service>_<fault>_<n>/metrics.parquet) if you ever use it.

Design choices (each one is a README talking point):
  * Features are SERVICE-AGNOSTIC: per metric type (cpu, mem, diskio, socket, latency, error, workload)
    we compute each service's robust z-shift after injection, then keep max / 2nd-max / sum, plus the
    full metric profile of the single most-anomalous service. Same feature vector for every system,
    so a model can be tested on a system it never saw (--loso).
  * Several windows per case (detection delay 0..N s) -> more rows, but CV is GROUPED BY CASE so
    windows of one incident never land in both train and test.
"""
import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FAULTS = ["cpu", "mem", "disk", "socket", "delay", "loss"]
KINDS = ["cpu", "mem", "diskio", "socket", "latency", "error", "workload"]


# ---------- loading ----------
def find_cases(root):
    """Yield (system, service, fault, rep, case_dir, metrics_file), one per case directory."""
    # First match wins per case. RE2 Zenodo zips ship BOTH simple_metrics.csv (72 cols: cpu, mem,
    # socket, latency-90, diskio, error, workload per service) and metrics.csv (~420 raw Prometheus
    # columns). The features are built on the simple set; raw-only cases are skipped in load_metrics.
    pref = ["simple_metrics.csv", "data.csv", "metrics.csv", "metrics.parquet", "metrics.json"]
    best = {}
    for name in pref:
        for p in glob.glob(os.path.join(root, "**", name), recursive=True):
            best.setdefault(os.path.dirname(p), p)
    for d, p in sorted(best.items()):
        base = os.path.basename(d)
        m = re.match(r"^(re\d[a-z]{2})_(.+)_([a-z]+)_(\d+)$", base)          # HF: re2ob_cartservice_cpu_1
        if m:
            yield m.group(1), m.group(2), m.group(3), int(m.group(4)), d, p
            continue
        parent = os.path.basename(os.path.dirname(d))                          # Zenodo: <svc>_<fault>/<n>
        m = re.match(r"^(.+)_([a-z]+)$", parent)
        if m and base.isdigit():
            system = os.path.basename(os.path.dirname(os.path.dirname(d))).lower().replace("-", "")
            yield system, m.group(1), m.group(2), int(base), d, p


def read_metrics_json(path):
    """RCAEval metrics.json: {metric_name: [[unix_ts, value], ...]} -> wide frame with a 'time' column."""
    import json
    with open(path) as fh:
        raw = {k: v for k, v in json.load(fh).items() if v}
    cols = {k: pd.Series({int(t): (np.nan if val is None else float(val)) for t, val in pts}) for k, pts in raw.items()}
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "time"
    return df.reset_index()


def load_metrics(path):
    if path.endswith(".json"):
        df = read_metrics_json(path)
    else:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("time.")])     # RE1-OB repeated header quirk
    if sum("container-" in c or "istio-" in c for c in df.columns) > 20:
        # raw Prometheus export (e.g. RE2 metrics.csv): metric names the feature code doesn't parse
        raise ValueError(f"raw Prometheus metrics only ({df.shape[1]} cols); no simple_metrics.csv")
    return df.replace([np.inf, -np.inf], np.nan).sort_values("time").ffill().fillna(0)


def inject_time(case_dir, cases_idx, key):
    f = os.path.join(case_dir, "inject_time.txt")
    if os.path.exists(f):
        with open(f) as fh:
            return int(fh.read().strip())
    if cases_idx is not None and key in cases_idx:
        return int(cases_idx[key])
    raise FileNotFoundError(f"no inject time for {case_dir}")


# ---------- features ----------
def kind_of(col):
    """'cartservice_latency-90' -> ('cartservice', 'latency'); unknown kinds -> None."""
    if "_" not in col:
        return None, None
    svc, k = col.split("_", 1)
    k = k.lower()
    if k.startswith("latency-50"):
        return None, None                      # keep p90 only, like the RCAEval baselines
    for kind in KINDS:
        if k.startswith(kind) or (kind == "diskio" and "disk" in k) or (kind == "error" and "err" in k):
            return svc.replace("-db", ""), kind
    return None, None


def robust_z(pre, post, z_floor=0.0, log_z=False):
    """Shift of the post-window median vs. the baseline, in robust (MAD) units.

    z_floor: minimum scale as a fraction of |baseline median|. Stops near-flat metrics
             (e.g. memory) from producing huge z for tiny changes. 0 = off (original).
    log_z:   compress with sign(z) * log1p(|z|) so a 30x and a 3000x shift stay distinct
             instead of both hitting the +-1000 clip.
    """
    med = pre.median()
    mad = (pre - med).abs().median() * 1.4826
    scale = mad.where(mad > 1e-9, pre.std())
    if z_floor > 0:
        scale = np.maximum(scale, z_floor * med.abs())
    scale = scale.replace(0, np.nan)
    z = ((post.median() - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0)
    if log_z:
        return np.sign(z) * np.log1p(z.abs())
    return z.clip(-1000, 1000)


def featurize(df, t_inj, off, pre_s, post_s, z_floor=0.0, log_z=False, norm_counts=False):
    """Baseline = pre_s seconds before injection; 'post' window starts `off` s late (detection delay)."""
    pre = df[(df.time < t_inj) & (df.time >= t_inj - pre_s)].drop(columns="time")
    post = df[(df.time >= t_inj + off) & (df.time < t_inj + off + post_s)].drop(columns="time")
    if len(pre) < 10 or len(post) < 5:
        return None
    z = robust_z(pre, post, z_floor=z_floor, log_z=log_z)
    by_kind, by_svc = defaultdict(dict), defaultdict(dict)
    for col, v in z.items():
        svc, kind = kind_of(col)
        if kind:
            by_kind[kind][svc] = v
            by_svc[svc][kind] = v
    big = np.log1p(3) if log_z else 3          # "clearly shifted" threshold, on the same scale as z
    feats = {}
    for kind in KINDS:
        vals = np.array(sorted(by_kind.get(kind, {}).values(), key=abs, reverse=True) or [0.0])
        feats[f"{kind}_max"] = vals[0]
        feats[f"{kind}_2nd"] = vals[1] if len(vals) > 1 else 0.0
        if norm_counts:
            # system-size independent: Train Ticket has many more services than Online Boutique,
            # so sums and counts across services would mostly encode WHICH system it is
            n = max(len(by_kind.get(kind, {})), 1)
            feats[f"{kind}_absmean"] = np.abs(vals).sum() / n
            feats[f"{kind}_frac_gt3"] = float((np.abs(vals) > big).sum()) / n
        else:
            feats[f"{kind}_abssum"] = np.abs(vals).sum()
            feats[f"{kind}_n_gt3"] = int((np.abs(vals) > big).sum())
    # profile of the most anomalous service (by total |z|) -> "what kind of anomaly is it?"
    top = max(by_svc, key=lambda s: sum(abs(x) for x in by_svc[s].values())) if by_svc else None
    tot = sum(abs(x) for x in by_svc.get(top, {}).values()) or 1.0
    for kind in KINDS:
        v = by_svc.get(top, {}).get(kind, 0.0)
        feats[f"top_{kind}"] = v
        feats[f"top_{kind}_share"] = abs(v) / tot
    feats["top_service"] = top
    return feats


def build(root, systems, offsets, pre_s, post_s, z_floor=0.0, log_z=False, norm_counts=False):
    idx = None
    cp = os.path.join(root, "cases.parquet")
    if os.path.exists(cp):
        c = pd.read_parquet(cp)
        tcol = next((x for x in c.columns if "inject" in x.lower()), None)
        if tcol is not None and "case" in c.columns:
            idx = dict(zip(c["case"], c[tcol], strict=False))
    rows = []
    for system, svc, fault, rep, d, p in find_cases(root):
        if systems and system not in systems:
            continue
        try:
            df = load_metrics(p)
            t0 = inject_time(d, idx, os.path.basename(d))
        except (OSError, ValueError, KeyError) as e:   # unreadable or malformed case: skip it, keep going
            print("skip", d, e, file=sys.stderr)
            continue
        for off in offsets:                     # simulate detection delay
            f = featurize(df, t0, off, pre_s, post_s, z_floor=z_floor, log_z=log_z,
                          norm_counts=norm_counts)
            if f:
                f.update(system=system, service=svc, fault=fault, case=f"{system}/{svc}_{fault}/{rep}", offset=off)
                rows.append(f)
    return pd.DataFrame(rows)


# ---------- evaluation ----------
def models():
    m = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=1, n_jobs=-1),
        "hist_gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=1),
    }
    try:
        from xgboost import XGBClassifier
        m["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.06, subsample=0.9,
                                     colsample_bytree=0.8, random_state=1, n_jobs=-1)
    except ImportError:
        pass
    return m


def evaluate(df, folds=5):
    X = df[[c for c in df.columns if c not in ("system", "service", "fault", "case", "offset", "top_service")]]
    labels = sorted(df.fault.unique())
    y = df.fault.map({lab: i for i, lab in enumerate(labels)}).values
    g = df.case.values
    k = min(folds, df.case.nunique())
    best = None
    print(f"\nGroupKFold(k={k}) by case — {len(df)} windows from {df.case.nunique()} cases, classes={labels}")
    for name, m in models().items():
        p = np.zeros_like(y)
        fold_f1 = []
        for tr, te in GroupKFold(k).split(X, y, g):
            p[te] = m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            fold_f1.append(f1_score(y[te], p[te], average="macro"))
        f1 = f1_score(y, p, average="macro")
        # ~45 cases per class: report the spread across folds, not just one pooled number
        print(f"  {name:14s} macroF1 per-fold mean={np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f} "
              f"(min {np.min(fold_f1):.3f}, max {np.max(fold_f1):.3f}) "
              f"| pooled={f1:.3f} acc={accuracy_score(y, p):.3f}")
        if name != "majority" and (best is None or f1 > best[1]):
            best = (name, f1, p)
    if best is None:
        raise RuntimeError("no model besides the majority baseline was evaluated")
    name, _, p = best
    # zero_division=0 is valid at runtime; sklearn's type hints only list the str options
    report = classification_report(
        y, p, target_names=labels, digits=3, zero_division=0  # pyright: ignore[reportArgumentType]
    )
    print(f"\nbest={name}\n{report}")
    print(pd.DataFrame(confusion_matrix(y, p), index=labels, columns=labels))
    # root-cause service as a free by-product: is the most anomalous service the injected one?
    hit = (df.top_service.str.replace("-", "") == df.service.str.replace("-", "")).mean()
    print(f"\nroot-cause service = most-anomalous service (no ML): {hit:.3f}")


def loso(df):
    print("\nLeave-one-system-out — the MAIN result (train on the other systems, test on the held-out one):")
    X = df[[c for c in df.columns if c not in ("system", "service", "fault", "case", "offset", "top_service")]]
    labels = sorted(df.fault.unique())
    y = df.fault.map({lab: i for i, lab in enumerate(labels)}).values
    candidates = [
        ("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("hist_gbm", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=1)),
    ]
    for name, make in candidates:
        scores = []
        for s in sorted(df.system.unique()):
            tr, te = (df.system != s).values, (df.system == s).values
            p = make().fit(X[tr], y[tr]).predict(X[te])
            scores.append(f1_score(y[te], p, average="macro"))
            print(f"  {name:9s} test={s:6s} macroF1={scores[-1]:.3f} acc={accuracy_score(y[te], p):.3f}")
        print(f"  {name:9s} mean over held-out systems: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--systems", nargs="*", default=None, help="e.g. re2ob re2ss re2tt")
    ap.add_argument("--offsets", nargs="*", type=int, default=[0, 30, 60, 90, 120], help="detection delays (s)")
    ap.add_argument("--pre", type=int, default=300, help="baseline window (s)")
    ap.add_argument("--post", type=int, default=120, help="post-detection window (s)")
    # Defaults chosen from the RE2-OB spike (Nov 3): XGBoost macro F1 0.65 -> 0.92, logreg 0.54 -> 0.85
    ap.add_argument("--z-floor", type=float, default=0.01,
                    help="min scale as a fraction of |baseline median|; 0 = off")
    ap.add_argument("--norm-counts", action="store_true",
                    help="use per-service means/fractions instead of sums/counts "
                         "(system-size independent; for leave-one-system-out)")
    ap.add_argument("--raw-z", action="store_true",
                    help="original features: no log compression, z clipped at +-1000 "
                         "(use with --z-floor 0 to reproduce the first spike)")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--out", default="features.parquet")
    a = ap.parse_args()
    log_z = not a.raw_z
    df = build(a.root, set(a.systems) if a.systems else None, a.offsets, a.pre, a.post,
               z_floor=a.z_floor, log_z=log_z, norm_counts=a.norm_counts)
    print(f"z settings: z_floor={a.z_floor} log_z={log_z} norm_counts={a.norm_counts}")
    if df.empty:
        sys.exit("no cases found — check the path/layout")
    try:
        df.to_parquet(a.out)
    except ImportError:                          # no pyarrow -> CSV
        df.to_csv(a.out.rsplit(".", 1)[0] + ".csv", index=False)
    print(df.groupby(["system", "fault"]).case.nunique().unstack(fill_value=0))
    evaluate(df)
    if a.loso and df.system.nunique() > 1:
        loso(df)