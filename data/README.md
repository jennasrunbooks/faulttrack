# Data

FaultTrack is trained on **RCAEval RE2** (Pham et al., *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*, WWW 2025 Companion), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- **Source:** Zenodo record [14590730](https://zenodo.org/records/14590730), [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730). We use the Zenodo copy, not the Hugging Face one.
- **You don't need the raw data** to run the notebook or `train.py`. They read the committed feature table in [`features/`](features/) (see [`features/ATTRIBUTION.md`](features/ATTRIBUTION.md)). You only need the steps below to rebuild that table from scratch.

## What we use

RE2 covers 3 microservice systems: Online Boutique (OB), Sock Shop (SS) and Train Ticket (TT). Each has 6 faults (`cpu`, `mem`, `disk`, `socket`, `delay`, `loss`) × 5 services × 3 repetitions, for **270 cases** in total (15 per fault per system).

| Suite | Download | md5 |
|---|---|---|
| RE2-OB | [RE2-OB.zip](https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1) (about 1.1 GB) | `b9e23f8842c404b396ffd2becff15de4` (verified Nov 3) |
| RE2-SS | [RE2-SS.zip](https://zenodo.org/records/14590730/files/RE2-SS.zip?download=1) | _TODO: copy from the Zenodo page and verify_ |
| RE2-TT | [RE2-TT.zip](https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1) | _TODO: copy from the Zenodo page and verify_ |

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

Known quirks (one case without `simple_metrics.csv`, `ts-` and `front-end` service names, `inf` values, a duplicated `time.1` column) are handled in `src/faulttrack/data.py`. See spec Section 3.3.

## Rebuild the feature table

```bash
make data && make features
```

`make features` writes `data/features/re2_features.parquet` (about 1,350 rows: 5 detection delays per case).
