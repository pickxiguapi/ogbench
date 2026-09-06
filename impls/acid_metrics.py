"""Numerically stable metrics for ACID action-feasibility diagnostics."""

from __future__ import annotations

import numpy as np


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float('nan')


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std()) if len(values) else float('nan')


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = average_ranks(scores)
    return float(
        (ranks[labels].sum() - positives * (positives - 1) / 2)
        / (positives * negatives)
    )


def correlation(left, right, *, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return float('nan')
    if rank:
        left = average_ranks(left)
        right = average_ranks(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def upper_tail_auc(scores, targets, *, quantile=0.75):
    """Measure whether high scores identify the upper tail of a real error."""
    if not 0.0 < quantile < 1.0:
        raise ValueError('quantile must be in (0, 1).')
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[finite]
    targets = targets[finite]
    if not len(targets):
        return float('nan'), float('nan'), 0
    threshold = float(np.quantile(targets, quantile))
    return binary_auc(targets >= threshold, scores), threshold, len(targets)


def risk_at_coverages(scores, risks, coverages=(0.25, 0.50, 0.75, 1.0)):
    """Return mean realized risk after retaining lowest-score events."""
    scores = np.asarray(scores, dtype=np.float64)
    risks = np.asarray(risks, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(risks)
    scores = scores[finite]
    risks = risks[finite]
    if not len(risks):
        return {float(coverage): float('nan') for coverage in coverages}
    ordered_risks = risks[np.argsort(scores, kind='mergesort')]
    output = {}
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError('coverages must be in (0, 1].')
        retained = max(1, int(np.ceil(coverage * len(ordered_risks))))
        output[float(coverage)] = float(ordered_risks[:retained].mean())
    return output
