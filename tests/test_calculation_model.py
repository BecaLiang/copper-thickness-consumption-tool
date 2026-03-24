import pytest

import numpy as np

from copper_usage.thickness_calculation import (
    Constraint,
    FitValue,
    FitValueCollection,
    DataColumns,
    PlainLinearModel,
    ThicknessCalculation,
    PlainLinearThicknessCalculation,
)
from copper_usage.sop_slicer import VCPLineRatioSlicer

from pydantic import ValidationError


def test_plain_linear_model():
    m = PlainLinearModel()
    assert m([1, 3], 2, 0.5) == 6.5
    assert m([2, -1], 1.5, 1) == -2.


def test_constraints():
    c1 = Constraint()
    assert c1.get() == [-np.inf, np.inf]
    c2 = Constraint(lower=0)
    assert c2.get() == [0, np.inf]
    c3 = Constraint(upper=100)
    assert c3.get() == [-np.inf, 100]
    c4 = Constraint(lower=-1, upper=1)
    assert c4.get() == [-1, 1]

    cdict = Constraint.init_dict_from_dict(
        {
            'A': {'lower': -10, 'upper': -5},
            'B': {'lower': 3},
            'C': {'upper': 1000},
        }
    )
    assert cdict['A'].get() == [-10, -5]
    assert cdict['B'].get() == [3, np.inf]


def test_fit_value_colletion():
    fvc = FitValueCollection.spawn_from_dict(
        {
            'time': 'time_column',
            'current': 'ampere',
        }
    )
    assert fvc.fit_parameters == ['time', 'current']
    assert 'time' in fvc
    assert FitValue('time', column='time_column') in fvc
    assert not FitValue('time', column='thickness_column') in fvc

    assert fvc.get_column('current') == 'ampere'

    fvc2 = FitValueCollection(
        fit_values=[
            FitValue('never'), 
            FitValue('gonna'), 
            FitValue('give'),
        ],
        _fit_parameters=['never', 'give'],
        )
    assert fvc2.fit_parameters == ['never', 'give']

    with pytest.raises(ValidationError):
        FitValueCollection(
            fit_values=[
                FitValue('never'), 
                FitValue('gonna'), 
                FitValue('give'),
            ],
            _fit_parameters=['you', 'up'],
        )


def test_data_columns():
    pass