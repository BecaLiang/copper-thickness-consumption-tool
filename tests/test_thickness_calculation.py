import pytest

import numpy as np
import pandas as pd

from copper_usage.data_columns import DataColumns, FitValue, FitValueCollection
from copper_usage.thickness_calculation import (
    PlainLinearThicknessCalculation,
    PlainLinearModel,
)
from copper_usage.sop_slicer import VCPLineRatioSlicer


@pytest.fixture
def tc_default():

    slicer = VCPLineRatioSlicer(
        is_vcp=True,
        ratio_min=3,
        ratio_max=5,
    )

    datacs = DataColumns(
        FitValueCollection(
            [
                FitValue('plating_time', column='time_pattern'),
                FitValue('current_density', column='current_pattern'),
                FitValue('target', column='minimal_thickness'),
            ]
        )
    )

    return PlainLinearThicknessCalculation(
        data_columns=datacs,
        data_slicer=slicer,
        calc=PlainLinearModel(),
        start_values=[],
    )


@pytest.fixture
def df_default():
    df = pd.DataFrame(
        {
            'time_pattern': [50, 55, 55, 60, 60, 60, 60, 65, 70, 70],
            'current_pattern': [18] * 5 + [20] * 5,
            'is_vcp': [True] * 10,
            'Ratio': [3.75, 4.25] * 5,
        }
    )
    df.loc[:, 'minimal_thickness'] = (
        0.05 
        * df.time_pattern 
        * df.current_pattern 
        + 10 
        + df.Ratio.map({3.75: -5, 4.25: 5})
    )
    return df


def test_data_cleaning(tc_default, df_default):

    df_broken = df_default.copy(deep=True)
    df_broken.loc[:, tc_default.data_columns.target_column] = np.nan
    with pytest.raises(AssertionError):
        tc_default.clean_data(df_broken)

    df_2 = df_default.copy()
    df_2.loc[1, 'time_pattern'] = np.nan
    df_2.loc[3, 'minimal_thickness'] = np.nan
    assert tc_default.clean_data(df_2).shape[0] == df_2.shape[0] - 2
    np.testing.assert_array_equal(
        tc_default.clean_data(df_2).index.values,
        np.array([0, 2, 4, 5, 6, 7, 8, 9])
    )

    df_3 = df_default.copy()
    df_3.loc[df_3.Ratio > 4, 'Ratio'] = 5.5
    assert tc_default.clean_data(df_3).Ratio.unique() == 3.75

    df_4 = df_default.copy()
    df_4.loc[:, 'is_vcp'] = False
    with pytest.raises(AssertionError):
        tc_default.clean_data(df_4)


def test_bulk_predict():
    pass

    


