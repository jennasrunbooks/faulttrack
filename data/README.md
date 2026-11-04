# Data

FaultTrack is trained on **RCAEval RE2** (Pham et al., *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*, WWW 2025 Companion), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- **Source:** Zenodo record [14590730](https://zenodo.org/records/14590730), [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730). We use the Zenodo copy, not the Hugging Face one.
- **You don't need the raw data** to run the notebook or `train.py`. They read the committed feature table in [`features/`](features/) (see [`features/ATTRIBUTION.md`](features/ATTRIBUTION.md)). You only need the steps below to rebuild that table from scratch.

## What we use

RE2 covers 3 microservice systems: Online Boutique (OB), Sock Shop (SS) and Train Ticket (TT). Each has 6 faults (`cpu`, `mem`, `disk`, `socket`, `delay`, `loss`) × 5 services × 3 repetitions, for **270 cases** in total (15 per fault per system).

| Suite | Download | md5 |
|---|---|---|
| RE2-OB | [RE2-OB.zip](https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1) (about 1.1 GB) | `b9e23f8842c404b396ffd2becff15de4` (verified Nov 3) |
| RE2-SS | [RE2-SS.zip](https://zenodo.org/records/14590730/files/RE2-SS.zip?download=1) (about 0.25 GB) | `bd747a8fc7c5be00c613e13fbf9dd74b` (verified Nov 4) |
| RE2-TT | [RE2-TT.zip](https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1) (about 2.8 GB) | `a7fbcd1ada406067dcc50771ae398408` (verified Nov 4) |

"Verified" means the md5 printed by `make data` equals the checksum Zenodo publishes for the file (`https://zenodo.org/api/records/14590730`).

The zips total about 4 GB, mostly logs and traces we don't use.

## Download

From the repo root, inside the devcontainer (needs `curl` and `unzip`):

```bash
make data
```

This runs `spike/download_rcaeval.sh` with `DEST=data/raw KEEP_ZIPS=0`. For each suite it:

1. downloads the zip from Zenodo (skipped if `data/raw/<suite>.zip` is already there),
2. prints its md5. **Compare it with the table above.**
3. extracts only the metrics files and `inject_time.txt` into `data/raw/<suite>/`; logs and traces stay in the zip,
4. deletes the zip.

`data/raw/` is gitignored. Never commit it, or the zips.

To fetch a single suite, call the script directly, for example `DEST=data/raw KEEP_ZIPS=0 bash spike/download_rcaeval.sh RE2-OB`.

## Case layout (confirmed)

After `make data`, each case is a folder `data/raw/RE2-XX/RE2-XX/<service>_<fault>/<n>/` (the suite name appears twice because the zip has its own top-level folder). It contains:

- **`simple_metrics.csv`:** 1,441 rows at 1 s intervals (24 minutes), with the fault injected at the midpoint. Per service: `cpu`, `mem`, `socket`, `workload`, `latency-50`, `latency-90`, `diskio`, `error`. **Features are built from this file only.**
- **`metrics.csv`:** the raw Prometheus export, about 420 columns, mostly cumulative counters. Not used in the graded version.
- **`inject_time.txt`:** the fault injection time, in Unix seconds.

The label comes from the folder name: `<service>_<fault>`, for example `cartservice_loss`.

Known quirks are handled in `src/faulttrack/data.py` (see spec Section 3.3):

- **An extra RE2-OB folder.** `checkoutservice_cpu/` has a fourth folder, `multi-source-data/`, with only `metrics.csv` and `inject_time.txt`. Its name isn't a repetition number, so it isn't a case and is ignored. RE2-OB has 90 numbered cases, all with `simple_metrics.csv`.
- **Rows with an empty `time`.** `checkoutservice_cpu/2` and `checkoutservice_mem/2` in RE2-OB end with 256 and 258 rows that have values but no `time`. Those rows are dropped (they can't be placed in time). Every detection window still has enough rows; only the delay-120 post window of `checkoutservice_cpu/2` is short (90 s of data instead of 120 s). A `time` cell holding text that isn't a number is still an error. See `docs/decisions.md`.
- `ts-` and `front-end` service names, `inf` values, and a duplicated `time.1` column.

## Rebuild the feature table

```bash
make data && make features
```

`make features` (`uv run python -m faulttrack.build`) takes about 15 s and writes two files to `data/features/`:

- **`re2_features.parquet`:** 1,350 rows (270 cases × 5 detection delays), sorted by `case_id` and `delay_s`. Identity columns `case_id`, `system`, `service`, `fault`, `delay_s` and `suspect_service` (the root-cause rule's pick), then seven `n_services_<kind>` counts (services with a column of that kind after the `-db` merge, for the `--norm-counts` experiment), then the features in `FEATURE_NAMES` order. Only `FEATURE_NAMES` are model inputs.
- **`re2_inventory.parquet`:** one row per case (270), with `status` (`ok` or the skip reason), row counts (`n_rows`, `n_untimed_rows`), metric columns in total and per kind, `nan_pct`, `inf_count`, `inject_pos` (where the injection falls in the file, 0 to 1) and `n_feature_rows`. The notebook uses it to show raw-data ranges and missing values without the raw download.

The build is deterministic. To check that a rebuild matches the committed files without overwriting them, run `uv run python -m faulttrack.build --check`. It compares the data with `pandas.testing.assert_frame_equal`, not the bytes, because parquet bytes can differ across pyarrow versions.
