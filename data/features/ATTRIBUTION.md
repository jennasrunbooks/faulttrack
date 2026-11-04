# Attribution

The two tables in this folder, `re2_features.parquet` (shift features) and `re2_inventory.parquet` (one row per case: sizes, column counts per metric kind, missing values), are derived from **RCAEval**:

> Pham et al. *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data.* WWW 2025 Companion.
> Dataset: Zenodo, [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730) (record [14590730](https://zenodo.org/records/14590730)).

- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The derived table is shared under the same license, with credit to the authors above.
- **Suites used:** RE2-OB, RE2-SS and RE2-TT (270 fault-injection cases).

## Changes made

The table is not a copy of the dataset. We changed it as follows:

1. **Metrics only.** From each RE2 zip we extracted only the metrics files and `inject_time.txt`. Logs and traces were not used. Features use `simple_metrics.csv` only.
2. **Derived shift features.** For each metric kind (`cpu`, `mem`, `diskio`, `socket`, `latency`, `error`, `workload`) and service, we computed a robust shift score: the change in median between a 300 s baseline and a 120 s window after detection, divided by a scale (1.4826 × MAD with a floor), then log-compressed. These scores are aggregated across services per kind, and summarized for the most anomalous service. No raw time series values are included.
3. **5 detection delays per case.** Each case gives 5 rows, with detection at 0, 30, 60, 90 and 120 s after the recorded injection time.
4. **Labels** (fault type and injected service) come from the case folder names. All 270 cases are used. An extra RE2-OB folder, `checkoutservice_cpu/multi-source-data/`, isn't a numbered case and is left out.
5. **Rows without a timestamp dropped.** Two RE2-OB files end with rows that have no `time` value; those rows are dropped before computing features.

The code that produces the tables is `src/faulttrack/features.py` and `src/faulttrack/build.py`, run with `make features`.
