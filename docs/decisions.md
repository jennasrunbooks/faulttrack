# Decisions

Design and evaluation decisions, newest last. Each entry says what was decided, the evidence, and what it changes. Read this before changing the evaluation, the features or the model choice.

## 2026-11-03: GO on RCAEval RE2

**Decision:** build FaultTrack on RCAEval RE2 with service-agnostic shift features and a tree model.

**Evidence** (from the go/no-go spike in `spike/`; these are not tuned final models):

| Run | Result (macro F1) |
|---|---|
| RE2-OB only, raw z | RF 0.704 ± 0.097 · XGB 0.647 · LR 0.543 · majority 0.048 |
| RE2-OB only, floor + log z | XGB **0.917 ± 0.076** · HGB 0.906 · RF 0.900 · LR 0.847 |
| **All 3 systems, grouped 5-fold** | **RF 0.828 ± 0.039** (min 0.769, max 0.875) · XGB 0.820 · HGB 0.819 · LR 0.673 · majority 0.048 |
| Root-cause rule (no ML), 3 systems | **0.853** (service hit rate) |
| Leave-one-system-out, HGB | OB 0.521 · SS 0.275 · TT 0.296 · mean **0.364 ± 0.112** |
| Leave-one-system-out, HGB, `--norm-counts` | mean 0.335 ± 0.118 (LR on TT 0.13 → 0.35) |

**Why GO:** every model beats the majority baseline (0.048) by a wide margin on grouped CV across all 3 systems, with a fold spread of about ± 0.04. Adding a scale floor and log compression to the shift score mattered more than the choice of model (LR on RE2-OB: 0.54 → 0.85).

## 2026-11-03: Headline is grouped CV across 3 systems; LOSO is a generalization section

**Decision:**
- The headline result is **grouped 5-fold CV across all 3 systems** (`GroupKFold`, `groups=case_id`), reported as macro F1 mean ± std (min/max) across folds.
- **Leave-one-system-out (LOSO)** is reported separately, in the "Generalizing to an unseen system" section, as a limitation with analysis.

**Why:** LOSO came in at **0.364 ± 0.112**, below the 0.45 threshold set for using it as the headline. `--norm-counts` didn't close the gap (0.335 ± 0.118). System size explains only part of it. The rest is domain shift: the same fault has a different magnitude and spread in each system.

**What it changes:** the README leads with the grouped-CV number and states the unseen-system gap next to it, rather than hiding it.

## 2026-11-04: Feature windows are anchored at the detection time

**Decision:** both feature windows are anchored at the detection time `t_det`: the baseline is `[t_det − 300 s, t_det)` and the post window is `[t_det, t_det + 120 s)`. `features.featurize()` takes only `t_det`. Training rows use `t_det` = injection time + a detection delay of 0, 30, 60, 90 or 120 s. Decided by Jenna on [issue #6](https://github.com/jennasrunbooks/faulttrack/issues/6#issuecomment-6639692891).

**Why:** the go/no-go spike anchored the baseline at the *injection* time and moved only the post window by the delay. That needs the injection time, which the API and real alerts never have, so training rows would not match what serving computes (train/serve skew). Anchoring both windows at `t_det` is what the API can reproduce from `detection_time` alone.

**What it changes:**
- Rows with delay 0 are unchanged from the spike.
- For delays of 30–120 s (1,080 of the 1,350 rows), the baseline now already includes some fault data, as it would for a real alert that fires late. The shifts get smaller as the delay grows.
- The headline (spike: RF 0.828 ± 0.039) may move when the feature table is rebuilt in #7. That is expected and is reported there with before/after mean ± std.

## 2026-11-04: A service and its `-db` keep the larger shift

**Decision:** when two columns map to the same (service, kind), which in RE2 means a service and its database (`carts` and `carts-db`) both having a column of the same kind, `features.featurize()` keeps the shift with the larger magnitude (on an exact tie, the larger signed value), for both the per-kind aggregates and the suspect service's profile. `data.kind_of()` now strips only a trailing `-db` (`removesuffix`), not `-db` anywhere in the name. Decided by Jenna in [review of PR #14](https://github.com/jennasrunbooks/faulttrack/pull/14#issuecomment-6640120673). This is the second intended difference from the spike, after the window anchoring above.

**Why:** the spike mapped both columns to the same `(service, kind)` key and assigned into a dict, so whichever column came later overwrote the other. That is a bug, not a modeling choice. A +8 step on `carts_cpu` with a flat `carts-db_cpu` after it gave `cpu_max` −0.14 instead of about 2.1, so the injected shift vanished from the features. The result also depended on column order.

**What it changes:** only cases with a `-db` service that shares a kind with its service. In RE2 that is expected to be Sock Shop (`re2ss`) only, so delay-0 rows for `re2ob` and `re2tt` should stay identical to the spike; #7 checks this on the real data. The effect on the headline and on the root-cause hit rate is reported in #7, when the feature table is rebuilt.

## 2026-11-04: Rows with an empty `time` are dropped

**Decision:** `data.clean_metrics()` drops rows whose `time` cell is empty (NaN or blank) before forward-filling. It still raises `ValueError` if the `time` column is missing, if a `time` cell holds text that isn't a finite number, or if no timed rows remain. The same code cleans API uploads, so an uploaded CSV with trailing untimed rows is accepted; `featurize()` still raises if the remaining rows don't cover the windows, which the API turns into a 422. The inventory records the count per case as `n_untimed_rows`. Decided by Jenna while building the feature table in #7. This is the third intended data-handling difference from the spike's code, after the two above, though unlike them it doesn't change any feature value.

**Why:** in RE2-OB, `checkoutservice_cpu/2` and `checkoutservice_mem/2` end with 256 and 258 rows that have metric values but no `time`. Their timed rows cover −720 s to +209 s and +283 s around the injection, enough for every detection window. Since PR #14, `clean_metrics()` raised on any missing `time`, which would have dropped both cases (268 cases). Rows without a time can't be placed in a window, so dropping them loses nothing. The spike kept these two cases: its sort put the untimed rows last, where no window reached them, so its feature values are the same as dropping the rows.

**What it changes:**
- The feature table has all **270 cases** (1,350 rows), the same cases as the spike.
- Spec Section 3.3 said `checkoutservice_cpu/4` has no `simple_metrics.csv`. The download has no `/4`; the 91st RE2-OB folder is `checkoutservice_cpu/multi-source-data/` (only `metrics.csv` and `inject_time.txt`). It isn't a numbered case, so `find_cases()` ignores it. The spec wording is proposed in #7.
