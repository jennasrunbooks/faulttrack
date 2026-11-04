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
