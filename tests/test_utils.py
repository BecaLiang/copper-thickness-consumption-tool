import pytest


from copper_usage.utils import (
    round_to_point_five, 
    deep_merge_two_dicts,
)


def test_deep_merge():
    dictA = {
        'A': 1,
        'B': {
            'C': 3,
            'D': 4,
            'E': 5,
        },
        'F': 6
    }
    dictB = {
        'B': {
            'F': 6
        },
        'G': 7,
    }
    dictC = {
        'A': 1,
        'B': {
            'C': 3,
            'D': 4,
            'E': 5,
            'F': 6,
        },
        'F': 6,
        'G': 7,
    }
    assert deep_merge_two_dicts(dictA, dictB) == dictC

    dictB = {
        'A': -99,
        'B': {
            'D': -77,
            'Z': 0,
        },
        'G': 10,
    }
    dictC = {
        'A': -99,
        'B': {
            'C': 3,
            'D': -77,
            'E': 5,
            'Z': 0,
        },
        'F': 6,
        'G': 10,
    }
    assert deep_merge_two_dicts(dictA, dictB) == dictC

def test_round_to_point_five():
    assert round_to_point_five(1.2) == 1
    assert round_to_point_five(1.4) == 1.5
    assert round_to_point_five(1.6) == 1.5
    assert round_to_point_five(1.8) == 2