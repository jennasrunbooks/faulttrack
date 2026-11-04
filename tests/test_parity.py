"""Training/serving parity: the fixture case featurized the way /predict/case
will do it must equal that case's rows in the committed feature table.

Runs offline: it reads only tests/fixtures/ and data/features/.
"""

from pathlib import Path

import pandas as pd
import pytest

from faulttrack.build import FEATURES_FILE, META_COLUMNS
from faulttrack.data import KINDS, clean_metrics, read_metrics
from faulttrack.features import DELAYS_S, FEATURE_NAMES, Featurized, featurize

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TABLE = ROOT / "data" / "features" / FEATURES_FILE
CASE_ID = "re2ob/productcatalogservice_cpu/1"


@pytest.fixture(scope="module")
def table_rows() -> pd.DataFrame:
    table = pd.read_parquet(TABLE)
    assert list(table.columns) == META_COLUMNS + FEATURE_NAMES
    rows = table[table["case_id"] == CASE_ID].set_index("delay_s")
    assert sorted(rows.index) == list(DELAYS_S)
    return rows


def _api_style_featurize(detection_time: int) -> Featurized:
    # the API gets an uploaded file object and a detection time
    with open(FIXTURES / "simple_metrics.csv", "rb") as upload:
        frame = clean_metrics(read_metrics(upload))
    return featurize(frame, detection_time)


@pytest.mark.parametrize("delay_s", DELAYS_S)
def test_fixture_matches_training_row(table_rows: pd.DataFrame, delay_s: int) -> None:
    inject_time = int((FIXTURES / "inject_time.txt").read_text().strip())
    result = _api_style_featurize(inject_time + delay_s)
    features = result.features
    row = table_rows.loc[delay_s]
    assert list(features) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert features[name] == pytest.approx(float(row[name]), abs=1e-9), name
    assert result.suspect_service == row["suspect_service"]
    for kind in KINDS:
        assert result.service_counts[kind] == row[f"n_services_{kind}"], kind
    assert row["fault"] == "cpu"
