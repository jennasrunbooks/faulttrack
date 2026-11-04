# Test fixture: one real RE2-OB case

`simple_metrics.csv` and `inject_time.txt` come from one RCAEval case. `tests/test_parity.py` uses them, and the README and REVIEWING.md `curl` examples post them to `/predict/case`.

| | |
|---|---|
| Source case | RE2-OB (Online Boutique), `productcatalogservice_cpu/1` |
| `case_id` | `re2ob/productcatalogservice_cpu/1` |
| True fault | `cpu`, injected into `productcatalogservice` |
| Injection time | `1705340542` (Unix seconds), in `inject_time.txt` |
| Columns | all 72 (`time` + 71 metric columns), unchanged |
| Rows | 780 of the original 1,441 |

**How it was trimmed.** The original file is about 1.18 MB, so rows were cut to keep it under 1 MB. The kept rows are the original CSV lines, byte for byte, with `time` in [injection − 420 s, injection + 360 s). That covers the 300 s baseline and 120 s post window for every detection delay used in training (0 to 120 s), with margin. Columns were not changed.

**Why this case.** The go/no-go spike's random forest classified all 5 of its windows correctly under grouped CV.

**Detection time for the API:** use the injection time, `1705340542`, which gives the `delay_s = 0` row of `data/features/re2_features.parquet`.

## Credit

From **RCAEval** (Pham et al., *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*, WWW 2025 Companion), Zenodo [doi:10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: rows trimmed as described above; values unchanged.
