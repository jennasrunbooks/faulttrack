"""train.py end to end on the committed feature table (no raw data, no network)."""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

import train
from faulttrack.features import FEATURE_NAMES

REQUIRED_RESULT_KEYS = [
    "fault_cv_macro_f1_mean",
    "fault_cv_macro_f1_std",
    "fault_cv_macro_f1_min",
    "fault_cv_macro_f1_max",
    "fault_cv_accuracy_mean",
    "fault_cv_roc_auc_ovr",
    "per_class",
    "confusion_matrix",
    "f1_by_delay",
    "loso",
    "suspect_hit_rate",
    "suspect_hit_rate_by_system",
    "baselines",
    "runbook_tree",
    "runbook_tree_cv_macro_f1",
    "sample_cases",
    "model_version",
    "n_rows",
    "n_cases",
    "generated_at",
]
REQUIRED_SPEC_KEYS = [
    "feature_names",
    "classes",
    "baseline_s",
    "post_s",
    "z_floor",
    "log_z",
    "model_name",
    "model_params",
    "model_version",
]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("train_out")
    train.main(out)
    return out, json.loads((out / "web" / "results.json").read_text())


def test_writes_all_three_artifacts(trained: tuple[Path, dict]) -> None:
    out, _ = trained
    for name in ("models/model.bin", "models/feature_spec.json", "web/results.json"):
        assert (out / name).is_file(), name


def test_feature_spec(trained: tuple[Path, dict]) -> None:
    out, results = trained
    spec = json.loads((out / "models" / "feature_spec.json").read_text())
    assert spec["feature_names"] == FEATURE_NAMES
    for key in REQUIRED_SPEC_KEYS:
        assert key in spec, key
    assert spec["model_version"] == results["model_version"]
    assert spec["model_version"].endswith("-rf")


def test_results_has_every_required_key(trained: tuple[Path, dict]) -> None:
    _, results = trained
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in results]
    assert not missing
    assert results["n_rows"] == 1350
    assert results["n_cases"] == 270
    assert len(results["baselines"]) == 2
    for key in ("mean", "std", "min", "max"):
        assert key in results["baselines"]["dummy"]
        assert key in results["baselines"]["rule"]
    assert 0 < results["fault_cv_macro_f1_mean"] <= 1
    assert results["baselines"]["dummy"]["mean"] < results["fault_cv_macro_f1_mean"]


def test_sample_cases_cover_each_fault_once(trained: tuple[Path, dict]) -> None:
    _, results = trained
    cases = results["sample_cases"]
    assert [c["true_fault"] for c in cases] == [
        "cpu",
        "mem",
        "disk",
        "socket",
        "delay",
        "loss",
    ]
    for case in cases:
        assert set(case) == {
            "case_id",
            "system",
            "true_fault",
            "predicted_fault",
            "probability",
            "suspect_service",
            "injected_service",
            "suspect_hit",
        }
        assert isinstance(case["suspect_hit"], bool)
        assert 0 < case["probability"] <= 1


def test_model_bin_predicts_one_row(trained: tuple[Path, dict]) -> None:
    out, _ = trained
    with open(out / "models" / "model.bin", "rb") as fh:
        model = pickle.load(fh)
    table = pd.read_parquet(train.FEATURES_PATH)
    row = table[FEATURE_NAMES].iloc[[0]]
    proba = model.predict_proba(row)
    assert proba.shape == (1, 6)
    assert model.predict(row)[0] in set(model.classes_)


def test_two_runs_give_the_same_results(
    trained: tuple[Path, dict], tmp_path: Path
) -> None:
    _, first = trained
    second = train.main(tmp_path)
    first = {k: v for k, v in first.items() if k != "generated_at"}
    second = {k: v for k, v in second.items() if k != "generated_at"}
    assert first == json.loads(json.dumps(second))


def test_sample_cases_rotate_through_all_systems(trained: tuple[Path, dict]) -> None:
    _, results = trained
    systems = [c["system"] for c in results["sample_cases"]]
    assert systems == ["re2ob", "re2ss", "re2tt"] * 2
