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

**Why:** in RE2-OB, `checkoutservice_cpu/2` and `checkoutservice_mem/2` end with 256 and 258 rows that have metric values but no `time`. Their timed rows cover −720 s to +209 s and +283 s around the injection. Every window has enough rows; the one short window is the delay-120 post window of `checkoutservice_cpu/2`, which has 90 s of data instead of 120 s (the spike saw the same). Since PR #14, `clean_metrics()` raised on any missing `time`, which would have dropped both cases (268 cases). Rows without a time can't be placed in a window, so dropping them loses nothing. The spike kept these two cases: its sort put the untimed rows last, where no window reached them, so its feature values are the same as dropping the rows.

**What it changes:**
- The feature table has all **270 cases** (1,350 rows), the same cases as the spike.
- Spec Section 3.3 said `checkoutservice_cpu/4` has no `simple_metrics.csv`. The download has no `/4`; the 91st RE2-OB folder is `checkoutservice_cpu/multi-source-data/` (only `metrics.csv` and `inject_time.txt`). It isn't a numbered case, so `find_cases()` ignores it. The spec wording is proposed in #7.

## 2026-11-05: Final model is a random forest (plan line 6)

**Decision:** the final classifier is a random forest with `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2` and `random_state=42`, defined in `faulttrack.models.final_model()`. Chosen in `notebook.ipynb` section 6 ([issue #16](https://github.com/jennasrunbooks/faulttrack/issues/16)); confirmed by Jenna in [review of PR #23](https://github.com/jennasrunbooks/faulttrack/pull/23#issuecomment-6671543660).

**Selection rule** (set in the issue before the results): the highest mean macro F1 from **nested** grouped CV; if a simpler model is within one fold std of the best, take the simpler one (simplest first: logistic regression, decision tree, random forest, XGBoost).

**Evidence** (grouped 5-fold CV by `case_id`, fold seed 42, macro F1 mean ± std (min/max) across folds):

| Model | Untuned | Tuned, nested CV |
|---|---|---|
| Dummy (most frequent) | 0.030 ± 0.002 (0.028/0.033) | — |
| Rule (biggest shift wins) | 0.277 ± 0.038 (0.212/0.318) | — |
| Logistic regression | 0.724 ± 0.027 (0.691/0.767) | 0.724 ± 0.024 (0.691/0.754) |
| Decision tree | 0.663 ± 0.046 (0.593/0.714) | 0.676 ± 0.049 (0.600/0.741) |
| **Random forest** | 0.800 ± 0.026 (0.776/0.846) | **0.800 ± 0.030 (0.778/0.859)** |
| XGBoost | 0.773 ± 0.020 (0.746/0.797) | 0.772 ± 0.018 (0.757/0.803) |

- **Why the random forest:** highest nested mean. XGBoost is the only other model within one std (0.772 ≥ 0.800 − 0.030) and is the more complex of the two, so the tie-break agrees. Logistic regression and the tree are more than one std behind.
- **Hyperparameters** come from whole-table tuning with the same grid on the same case-grouped folds (`cv=list(grouped_splits(df))`); that search's best score was 0.799. The nested score (0.800 ± 0.030) is the unbiased estimate of the tuning procedure.
- **Headline for `final_model()`:** `grouped_cv(final_model(), df)` gives **0.799 ± 0.028 (min 0.775, max 0.850)**, accuracy 0.807 ± 0.025. That is 0.001 below the sanity check in #7 (untuned RF, 0.800 ± 0.026), so it is not a metrics change. `tests/test_models.py` pins it.
- **Seed sensitivity:** with fold seeds 0–4 the mean is 0.787, 0.803, 0.823, 0.797 and 0.803 (mean 0.803, range 0.036). The headline stays seed 42; differences under about 0.03 between models aren't meaningful.
- **Tuning barely matters:** no model moved by more than 0.013 between untuned and tuned (nested).
- **Also reported in the notebook:** macro one-vs-rest ROC AUC 0.974; macro F1 0.787–0.806 at each detection delay from 0 to 120 s; main confusions delay ↔ loss (56/54 rows) and cpu ↔ socket (40/47); a depth-3 runbook tree at 0.570 ± 0.067. The suspect-service rule's hit rate is 0.900 per alert window, reported separately as a rule.

**Headline (decided by Jenna in the same review):** quote **0.799 ± 0.028**, the score of the exact model `train.py` reproduces, with the nested **0.800 ± 0.030** next to it. Both are shown because the shipped hyperparameters were chosen on the same seed-42 folds that score them, so the nested number is the unbiased estimate of the tune-then-fit procedure.

**No XGBoost early stopping** (spec 5.1 listed it): XGBoost isn't the final model, the grid tunes `n_estimators` directly, and early stopping would need its own grouped validation split inside each fold. The spec is updated to match (maintainer change).

**What it changes:** `train.py` (plan line 8) fits `final_model()` on all rows. The README's results section quotes the headline as above, with the seed range next to it.

**Nested-tuning note:** `grouped_cv` now passes `groups=` (the training fold's `case_id`) to a `GridSearchCV` or `RandomizedSearchCV`, so a search with `cv=GroupKFold(5)` is nested and grouped. `cv=list(grouped_splits(df))` is for whole-table tuning only: inside nested CV those indices would refer to the full table and leak outer-test cases into the inner search.

## 2026-11-05: Leave-one-system-out re-measured on the committed table (plan line 7)

**Result:** leave-one-system-out (LOSO; train on two systems, test on the third) for `final_model()` gives **macro F1 0.568 ± 0.114** across the 3 held-out systems: OB 0.655, SS 0.643, TT 0.407. It is in `notebook.ipynb` section 8.1 ([issue #17](https://github.com/jennasrunbooks/faulttrack/issues/17)). The mean ± std is across 3 systems, not folds.

| Model | LOSO: OB | LOSO: SS | LOSO: TT | LOSO mean ± std | Grouped CV (headline) |
|---|---|---|---|---|---|
| Random forest (`final_model()`) | 0.655 | 0.643 | 0.407 | **0.568 ± 0.114** | 0.799 ± 0.028 |
| Logistic regression (untuned) | 0.583 | 0.606 | 0.303 | 0.497 ± 0.138 | 0.724 ± 0.027 |
| Rule (biggest shift wins) | 0.199 | 0.279 | 0.288 | 0.255 ± 0.040 | 0.277 ± 0.038 |

- **Higher than the spike's 0.364 ± 0.112.** HGB, the spike's LOSO model, gets 0.553 ± 0.065 on the committed table, so the change comes from the features rebuilt after the spike (the window anchoring and `-db` entries above), not from the model. The spike's feature version can't be rebuilt without `data/raw/`, so the cause isn't split further.
- **Tuning isn't the reason:** `FINAL_PARAMS` were tuned on all three systems, but the untuned forest scores 0.559 ± 0.118.
- **Per system:** the final model's grouped-CV out-of-fold rows score OB 0.769, SS 0.759, TT 0.896 (pooled). Held out, OB and SS lose about 0.11 each and TT loses 0.489.
- **Why:** partly system size (TT has 68 services against 11, and the `<kind>_abssum` and `<kind>_n_gt3` features add up over services); mainly domain shift. In TT a memory fault shifts `diskio` (median `diskio_max` 6.91, against 0.07 and 0.32 in OB and SS), so with TT held out all 75 memory rows are predicted as disk. delay ↔ loss is the main confusion across systems (190 of 1,350 rows, against 110 under grouped CV). Only 31 LOSO errors mix a resource fault with a network fault.

**What it changes:** nothing in the code. The 2026-11-03 entry set a 0.45 LOSO threshold for making LOSO the headline, and 0.568 is now above it. **The headline stays grouped CV pending Jenna's decision** (issue #17 hand-off; the default is no change). The README limitation text is at the end of notebook section 8.5.

**Decision (Jenna, 2026-11-06):** the headline stays grouped CV; the README's first Results sentence gives the LOSO number next to it. The 0.45 rule from 2026-11-03 is retired.

**Why the rule was retired rather than applied:** it was set before these results, so the reason is recorded here. LOSO is a mean over only 3 systems, from 0.407 (TT) to 0.655 (std 0.114, against 0.028 across the 5 grouped-CV folds), which is too unstable for the single headline. It also answers a different question: the API's `cv_macro_f1` describes fault type on systems like those the model was trained on, and for a new system the README says to retrain on that cluster's labeled incidents. LOSO isn't hidden: it's in the README's first Results sentence and next to the headline on the results page.

## 2026-11-05: `--norm-counts` gives a small LOSO gain; not adopted (plan line 7)

**Experiment:** replace `<kind>_abssum` and `<kind>_n_gt3` with `<kind>_absmean` and `<kind>_frac_gt3` (divided by `max(n_services_<kind>, 1)`), rebuilt in memory from the committed table and checked against `featurize(norm_counts=True)` on the fixture case. Notebook section 8.3.

| `final_model()` | LOSO mean ± std (seed 42) | Grouped CV (fold seed 42) | LOSO mean, seeds 0–4 | Grouped CV mean, seeds 0–4 |
|---|---|---|---|---|
| Headline features | 0.568 ± 0.114 | 0.799 ± 0.028 (0.775/0.850) | 0.572 | 0.803 |
| Norm-counts | 0.600 ± 0.095 | 0.775 ± 0.023 (0.746/0.814) | 0.596 | 0.798 |

- **LOSO:** +0.013 to +0.039 at every forest seed (mean +0.024), mostly on TT (+0.058 at seed 42), where it cuts the cpu → socket errors from 54 to 11 rows. It doesn't fix TT's mem → disk errors (74 of 75), and it moves loss → delay from 45 to 66 rows.
- **Grouped CV:** −0.005 on average over fold seeds 0–4 (from −0.013 to +0.002); the −0.024 at seed 42 is mostly that one fold split.
- **Logistic regression:** LOSO falls (0.497 → 0.462), although TT rises as in the spike (0.303 → 0.370).

**Decision:** not adopted; the headline features stay. The gain is consistent but small (under the 0.05 that `AGENTS.md` treats as a metrics change), leaves LOSO about 0.20 below grouped CV, and adopting it would mean regenerating the feature table, `models/` and the API's feature spec 11 days before submission. If Jenna wants it, it goes in a follow-up issue (`features.py`, feature table and `final_model()` are out of scope for #17).

## 2026-11-05: Relative (per-system standardized) features don't help; not adopted (plan line 7)

**Experiment:** standardize every feature within each system (mean and std, `ddof=0`) using only that system's rows and no labels: each training system with its own rows, the held-out system with its own rows when it is scored. Time-boxed to half a day. Notebook section 8.4.

| Model | LOSO mean ± std, headline | LOSO mean ± std, relative |
|---|---|---|
| Random forest (`final_model()`) | 0.568 ± 0.114 | 0.571 ± 0.120 (OB 0.706, SS 0.594, TT 0.414) |
| Logistic regression | 0.497 ± 0.138 | 0.533 ± 0.073 (TT 0.303 → 0.431) |

- With the held-out system's statistics from only 18 random cases (20%, 5 draws), LOSO is 0.574 (0.555–0.590) for the forest and 0.529 for logistic regression, so the result doesn't depend on having many windows. The fault mix of those windows isn't tested (RE2 is balanced by design).
- Grouped CV for the forest has the same mean ± std (0.799 ± 0.028; fold min/max 0.761/0.844 against 0.775/0.850), with each system's statistics taken over all its rows, test folds included (unlabeled). TT's mem → disk errors stay at 75 of 75.

**Decision:** not adopted. No change to the final model's LOSO score; the gain for logistic regression doesn't bring it up to the forest. The experiment stopped inside the time box.
