import pytest

import numpy as np

from numpy.testing import assert_array_almost_equal
from scipy.stats import norm
from copper_usage.inverter import PlainSquareScore


def test_plain_score():
    pss = PlainSquareScore()
    f_pred = lambda X: np.array(
            [0 for _ in X]
        )
    arr_test = pss(
        predict=f_pred,
        xs=np.arange(-2, 2.1, 0.5),
        y=0.05,
        minreq=1,
        sigma_v=1,
    )
    pred_truth = norm.cdf(
        x=1,
        loc=f_pred(np.arange(-2, 2.1, 0.5)),
        scale=1,
    )
    arr_truth=(pred_truth - 0.05) ** 2
    assert_array_almost_equal(arr_test, arr_truth)


def test_error_model():
    pass