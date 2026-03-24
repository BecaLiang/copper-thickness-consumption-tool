import pytest

import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from copper_usage.sop_slicer import VCPLineRatioSlicer

from pydantic import ValidationError


@pytest.fixture
def df_test():
    return pd.DataFrame(
        {
            'Ratio': list(np.arange(1, 9.51, 0.5)),
            'is_vcp': [True, False] * 9,
        }
    )


def test_vcplineratio_basic_construction():
    
    with pytest.raises(ValidationError):
        VCPLineRatioSlicer(is_vcp=False, ratio_min=1, ratio_max=0)
    with pytest.raises(ValidationError):
        VCPLineRatioSlicer(is_vcp=False, ratio_min=-1, ratio_max=3)
    
    # basically assert that this does NOT raise anything
    _ = VCPLineRatioSlicer(is_vcp=False, ratio_min=1, ratio_max=3)
    _ = VCPLineRatioSlicer(is_vcp=True, ratio_min=0, ratio_max=3)
    _ = VCPLineRatioSlicer(is_vcp=True, ratio_min=1, ratio_max=np.inf)
    

def test_vcplineratio_df_slicing(df_test):
    
    sl1 = VCPLineRatioSlicer(is_vcp=True, ratio_min=0.5, ratio_max=1.25)
    assert_frame_equal(
        sl1.slice_data(df_test), 
        pd.DataFrame({'Ratio': [1.], 'is_vcp': [True]}),
    )
    
    sl2 = VCPLineRatioSlicer(is_vcp=False, ratio_min=0.5, ratio_max=1.51)
    assert_frame_equal(
        sl2.slice_data(df_test), 
        pd.DataFrame({'Ratio': [1.5], 'is_vcp': [False]}, index=[1]),
    )

    sl3 = VCPLineRatioSlicer(is_vcp=True, ratio_min=2.0, ratio_max=3.5)
    assert_frame_equal(
        sl3.slice_data(df_test), 
        pd.DataFrame({'Ratio': [2.0, 3.0], 'is_vcp': [True, True]}, index=[2, 4]),
    )

    sl4 = VCPLineRatioSlicer(is_vcp=False, ratio_min=2.0, ratio_max=3.5)
    assert_frame_equal(
        sl4.slice_data(df_test), 
        pd.DataFrame({'Ratio': [2.5], 'is_vcp': [False]}, index=[3]),
    )


def test_vcplineratio_df_initialization():

    slices = VCPLineRatioSlicer.initialize_slices({'slice_boarders': [1, 2, 3]})
    assert all([s1 < s2 for s1, s2 in zip(slices[:-1], slices[1:])])
    assert len(slices) == 4
    assert not slices[0].is_vcp
    assert slices[2].is_vcp
    assert slices[0].ratio_min == 1
    assert slices[2].ratio_min == 1
    assert slices[1].ratio_max == 3
    assert slices[3].ratio_max == 3

    slices = VCPLineRatioSlicer.initialize_slices(
        {
            'slice_boarders': [1, 2, 3], 
            'overflow': True,
        }
    )
    assert len(slices) == 6
    assert slices[2].ratio_max == np.inf

    slices = VCPLineRatioSlicer.initialize_slices(
        {
            'slice_boarders': [1, 2, 3], 
            'underflow': True,
        }
    )
    assert len(slices) == 6
    assert slices[3].ratio_min == 0

    slices = VCPLineRatioSlicer.initialize_slices(
        {
            'slice_boarders': [0, 1, 2, 3], 
            'underflow': True,
        }
    )
    assert len(slices) == 6


def test_being_in_charge():
    slice1 = VCPLineRatioSlicer(
        is_vcp=True,
        ratio_min=1,
        ratio_max=2, 
    )
    assert slice1.contains(1.5)
    assert slice1.contains(1)
    assert not slice1.contains(2)
    assert not slice1.contains(2.3)
    assert not slice1.contains(0.5)
    assert not slice1.contains(-0.5)
    assert slice1.is_in_charge(
        is_vcp=True,
        Ratio=1.5,
    )
    assert not slice1.is_in_charge(
        is_vcp=False,
        Ratio=1.5,
    )
    assert not slice1.is_in_charge(
        is_vcp=True,
        Ratio=3.5,
    )
    slice2 = VCPLineRatioSlicer(
        is_vcp=False,
        ratio_min=5,
        ratio_max=np.inf, 
    )
    assert slice2.contains(100)
    assert not slice2.contains(3)
    assert not slice2.is_in_charge(
        is_vcp=False,
        Ratio=3.5,
    )
    assert slice2.is_in_charge(
        is_vcp=False,
        Ratio=6.5,
    )