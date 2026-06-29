import pytest

import numpy as np
import pandas as pd

from copper_usage.model import Model
from copper_usage.feature_containers import BoardFeatureContainer
from copper_usage.data_columns import DataColumns, FitValueCollection
from copper_usage.thickness_calculation import PlainLinearThicknessCalculation, PlainLinearModel
from copper_usage.sop_slicer import VCPLineRatioSlicer


@pytest.fixture
def data_column_def():
    return DataColumns.init_from_config(
        {
            c: c for c in FitValueCollection.default_fit_parameters()
        }
    )


def test_feature_extraction(monkeypatch):

    column_dict = {
        'requirement_column': 'required_thickness',
        'vcp_column': 'is_vcp',
        'ratio_column': 'Ratio',
        'board_thickness': 'board_thickness',
    }

    monkeypatch.setattr(Model, "build_combined_column_dict", lambda self: column_dict)

    m1 = Model(thickness_calculations=[])
    df = pd.DataFrame(
        {
            'required_thickness': [18, 20, 22],
            'is_vcp': [True, False, True],
            'Ratio': [3, 4, 5.5],
            'board_thickness': [1, 1, 1.2],
        },
    )
    test_list = m1.extract_board_feature_list(df, margin=0.05)
    truth_list = [
        BoardFeatureContainer(
            margin=0.05,
            required_thickness=18,
            is_vcp=True,
            Ratio=3,
            board_thickness=1,
        ),
        BoardFeatureContainer(
            margin=0.05,
            required_thickness=20,
            is_vcp=False,
            Ratio=4,
            board_thickness=1,
        ),
        BoardFeatureContainer(
            margin=0.05,
            required_thickness=22,
            is_vcp=True,
            Ratio=5.5,
            board_thickness=1.2,
        ),
    ]

    assert truth_list == test_list


def test_select_calculator_idx(data_column_def):
    
    tcs = [
        PlainLinearThicknessCalculation(
            data_columns=data_column_def,
            data_slicer=VCPLineRatioSlicer(
                is_vcp=True, 
                ratio_min=rmin, 
                ratio_max=rmax,
            ),
            calc=PlainLinearModel(),
            start_values=[1, 1, 1, 1],
        ) for rmin, rmax in zip(
            [0, 2, 3, 5],
            [2, 3, 4, np.inf],
        )
    ]

    m1 = Model(tcs)

    assert m1._select_calculator_idx(Ratio=1, is_vcp=True) == 0
    assert m1._select_calculator_idx(Ratio=3.5, is_vcp=True) == 2
    assert m1._select_calculator_idx(Ratio=9, is_vcp=True) == 3
    with pytest.raises(KeyError):
        m1._select_calculator_idx(Ratio=1, is_vcp=False)
    with pytest.raises(KeyError):
        m1._select_calculator_idx(Ratio=4.5, is_vcp=True)
    assert m1._select_calculator_idx(Ratio=1, is_vcp=False, raise_if_missing=False) is None
