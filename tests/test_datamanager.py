import pytest

import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from copper_usage.datamanager import EnforceFloat, Pretransformations


def test_enforce_float():
    ef1 = EnforceFloat(columns='a')
    df1 = pd.DataFrame({'a': ['1', '2', ' '], 'b': [1, 2, 3]})
    
    assert_frame_equal(
        ef1.apply(df1),
        pd.DataFrame({'a': [1., 2., np.nan], 'b': [1, 2, 3]}),
    )
    
    ef2 = EnforceFloat(columns='b')
    df2 = pd.DataFrame({'a': ['1', '2', ' '], 'b': [1, 2, 3]})
    assert_frame_equal(
        ef2.apply(df2),
        pd.DataFrame({'a': ['1', '2', ' '], 'b': [1., 2., 3.]}),
    )


def test_init_pretrafos():
    
    p1 = Pretransformations.init_from_config(
        cfg={
            'enforce_float': ['a', 'b'],
        },
    )
    
    df1 = pd.DataFrame(
        {
            'a': ['1', '2', ' '], 
            'b': [1, 2, 3],
            'c': ['a', 'b', 'c'],
        },
    )

    assert_frame_equal(
        p1.apply(df1),
        pd.DataFrame(
           {
                'a': [1., 2., np.nan], 
                'b': [1., 2., 3.],
                'c': ['a', 'b', 'c'],
            },
        ),
    )