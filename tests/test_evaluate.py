import math

import numpy as np
import pytest

pytest.importorskip("torch")
from src import evaluate as ev  # noqa: E402


def test_perfect_predictions():
    m = ev.compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], thr=0.5)
    assert m["accuracy"] == m["sensitivity"] == m["specificity"] == 1.0
    assert m["auroc"] == 1.0


def test_known_confusion_matrix():
    y = [0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.6, 0.2, 0.9, 0.4, 0.8, 0.7]
    m = ev.compute_metrics(y, p, thr=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 3)
    assert m["sensitivity"] == 0.75
    assert m["precision"] == 0.75
    assert m["specificity"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["accuracy"] == pytest.approx(5 / 7, abs=1e-3)


def test_best_threshold_separates_separable_data():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    thr = ev.best_threshold(y, p)
    assert 0.2 < thr <= 0.8


def test_safe_auc_single_class_is_nan():
    assert math.isnan(ev.safe_auc([0, 0, 0], [0.1, 0.2, 0.3]))
