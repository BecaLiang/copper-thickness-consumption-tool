import pytest

import numpy as np

from copper_usage.data_columns import (
    Constraint, 
    DataColumns, 
    FitValue, 
    FitValueCollection,
)
from pydantic import ValidationError


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
    
    dc1 = DataColumns(
        fit_values=FitValueCollection(
            [
                FitValue('thickness', column='board thickness'),
                FitValue('duration', column='plating time'),
                FitValue('current', column='ampere value'),
                FitValue('target', column='minimal pattern thickness'),
            ]
        ),
    )
    
    assert dc1.all_columns == ['board thickness', 'plating time', 'ampere value', 'minimal pattern thickness']
    dc1.set_fit_parameters(['thickness', 'duration'])
    assert dc1.fit_columns == ['board thickness', 'plating time']
    assert dc1.relevant_columns == ['board thickness', 'plating time', 'minimal pattern thickness']
    assert dc1.target_column == 'minimal pattern thickness'
    assert dc1.get_boundaries() is None

    dc2 = DataColumns(
        fit_values=FitValueCollection(
            [
                FitValue('thickness', column='board thickness'),
                FitValue('duration', column='plating time'),
                FitValue('current', column='ampere value'),
                FitValue('the_target', column='minimal pattern thickness'),
            ]
        ),
    )

    with pytest.raises(KeyError):
        dc2.target_column

    with pytest.raises(KeyError):
        dc2.relevant_columns


def test_datacolumns_init():

    cfg = {
        'thickness': 'board thickness',
        'duration': 'plating time',
        'current': 'ampere value',
        'target': 'minimal pattern thickness',
        'constraints': {
            'thickness': {'lower': 0},
            'duration': {'lower': 0},
            'current': {'lower': -10, 'upper': 10},
        },
        'fit_parameters': ['thickness', 'duration'],
    }

    dc1 = DataColumns.init_from_config(cfg)
    assert dc1.fit_columns == ['board thickness', 'plating time']
    assert dc1.relevant_columns == ['board thickness', 'plating time', 'minimal pattern thickness']
    assert dc1.target_column == 'minimal pattern thickness'

    dc2 = DataColumns.init_from_config(
        {
            'plating_time': 'plating time',
            'current_density': 'all of the amperes',
            'board_thickness': 'board thickness mean',
            'target': 'big profit',
            'fit_parameters': ['board_thickness', 'plating_time']
        }, 
        use_default=True,
    )
    assert dc2.fit_columns == ['board thickness mean', 'plating time']
    assert dc2.relevant_columns == ['board thickness mean', 'plating time', 'big profit']