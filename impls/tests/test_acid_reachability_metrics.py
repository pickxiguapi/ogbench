import numpy as np
import pytest

from acid_metrics import (
    binary_auc,
    correlation,
    grouped_risk_at_coverages,
    grouped_upper_tail_auc,
    risk_at_coverages,
    upper_tail_auc,
)


def test_binary_auc_handles_ties_and_direction():
    labels = np.asarray([False, False, True, True])
    assert binary_auc(labels, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert binary_auc(labels, [3.0, 2.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert binary_auc(labels, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_first_block_calibration_metrics_have_expected_direction():
    acid_scores = np.asarray([0.1, 0.2, 0.3, 0.4])
    realized_errors = np.asarray([1.0, 2.0, 3.0, 4.0])
    auc, threshold, count = upper_tail_auc(
        acid_scores, realized_errors, quantile=0.75
    )
    assert auc == pytest.approx(1.0)
    assert threshold == pytest.approx(3.25)
    assert count == 4
    assert correlation(acid_scores, realized_errors) == pytest.approx(1.0)
    assert correlation(acid_scores, realized_errors, rank=True) == pytest.approx(1.0)


def test_risk_coverage_retains_lowest_acid_scores_and_filters_nan():
    risks = risk_at_coverages(
        scores=[0.3, np.nan, 0.1, 0.2],
        risks=[3.0, 100.0, 1.0, 2.0],
    )
    assert risks[0.25] == pytest.approx(1.0)
    assert risks[0.50] == pytest.approx(1.5)
    assert risks[0.75] == pytest.approx(2.0)
    assert risks[1.0] == pytest.approx(2.0)


def test_invalid_calibration_quantile_is_rejected():
    with pytest.raises(ValueError):
        upper_tail_auc([0.1], [1.0], quantile=1.0)


def test_grouped_metrics_remove_between_state_difficulty_confound():
    groups = np.repeat([0, 1], 4)
    scores = np.tile([0.1, 0.2, 0.3, 0.4], 2)
    targets = np.asarray([1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0])
    auc, labels, thresholds = grouped_upper_tail_auc(
        scores, targets, groups, quantile=0.75
    )
    assert auc == pytest.approx(1.0)
    np.testing.assert_array_equal(
        labels, [False, False, False, True, False, False, False, True]
    )
    assert thresholds == {'0': pytest.approx(3.25), '1': pytest.approx(103.25)}

    risks = grouped_risk_at_coverages(scores, targets, groups)
    assert risks[0.25] == pytest.approx(51.0)
    assert risks[0.50] == pytest.approx(51.5)
    assert risks[1.0] == pytest.approx(52.5)
