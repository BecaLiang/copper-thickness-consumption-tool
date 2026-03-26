import pytest

from copper_usage.thickness_calculation import (
    PlainLinearModel,
    PlainLinearModelWithBoard,
    PlainLinearModelWithBoardAndSpray,
    ThicknessCalculation,
    PlainLinearThicknessCalculation,
)


def test_plain_linear_model():
    m = PlainLinearModel()
    assert m([1, 3], 2, 0.5) == 6.5
    assert m([2, -1], 1.5, 1) == -2.


def test_plain_linear_with_board():
    m = PlainLinearModelWithBoard()
    assert m([1, 2, 0.5], slope=3, bthick=1, offset=-1) == 5.5
    assert m([2, 1, 1], slope=0.5, bthick=2, offset=1) == 4.


def test_plain_linear_with_board_and_spray():
    m = PlainLinearModelWithBoardAndSpray()
    assert m([1, 2, 1, 0.5], slope=3, bthick=1, pspray=-2, offset=-1) == 5.
    assert m([2, 1, 0, 1], slope=0.5, bthick=2, pspray=1, offset=1) == 3.