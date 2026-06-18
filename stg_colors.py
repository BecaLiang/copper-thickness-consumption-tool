from __future__ import annotations


import numbers as nbr
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from collections import UserDict

from utils import make_int


_stg_all_colors = {
    1: {
        'blue': '#002D74',
        'gray': '#606163',
        'red': '#E10810',
        'black': '#111921',
        'white': '#FFFFFF',
        'yellow': '#FAC905',
        'green': '#228B22',
    },
    2: {
        'blue': '#0051D1',
        'gray': '#898A8D',
        'red': '#E9464C',
        'black': '#111921',
        'white': '#FFFFFF',
        'yellow': '#FFF440', 
        'green': '#4AD24A',
    },
    3: {
        'blue': '#3F89FF',
        'gray': '#B2B3B7',
        'red': '#F08488',
        'black': '#111921',
        'white': '#FFFFFF',
    },
    4: {
        'blue': '#A1C5FF',
        'gray': '#D4D6DB',
        'red': '#F8C1C3',
        'black': '#111921',
        'white': '#FFFFFF',
    },
}


stg_base_colors = _stg_all_colors[1].copy()


def stg_color(color: str, depth: int=1) -> str:
    try:
        colors = _stg_all_colors[depth]
    except KeyError:
        raise KeyError(
            f"{depth} is not a valid shade-key."
            f"Must be one of {list(_stg_all_colors.keys())}"
        )
    try:
        return colors[color]
    except KeyError:
        raise KeyError(f"{color} is not a valid color at shade-depth {depth}."
                       f"Can be one of {list(colors.keys())}")


@dataclass
class ColorSet:    

    depth: int

    blue: str
    gray: str
    red: str
    yellow: str | None = None
    green: str | None = None
    white: str = '#FFFFFF'
    black: str = '#111921'

    @classmethod
    def init(cls, depth: int=1) -> ColorSet:
        depth_key = make_int(depth)
        try:
            obj = cls(depth=depth_key, **_stg_all_colors[depth_key])
        except KeyError:
            raise KeyError(f"{depth} is not a valid shade-key."
            f"Must be one of {list(_stg_all_colors.keys())}")

        return obj

    @cached_property
    def valid_colors(self):
        return [color for color, value in self.__dict__.items() 
                if isinstance(value, str)]

    def get_color(self, color: str) -> str:
        if not isinstance(color, str):
            raise TypeError(f'A color name must be a string; {color} is {type(color)}')
        return getattr(self, color)


class ColorSetRavelled:
    def __init__(
        self,
        color_sets: list[ColorSet],
        depth_delim: str=':',
    ):
        self.color_sets = color_sets
        self.depth_delim = depth_delim

    @classmethod
    def init(
        cls, 
        depth_limit: list[int]=None,
        depth_delim: str=':',
    ) -> ColorSetRavelled:
        if depth_limit:
            color_sets = {depth: ColorSet.init(depth) for depth in depth_limit}
        else:
            color_sets = {
                depth: ColorSet.init(depth) 
                for depth in _stg_all_colors.keys()
            }
        return cls(color_sets, depth_delim=depth_delim)

    @staticmethod
    def split_color_name(colorname: str, delim: str) -> tuple[str, int]:
        parts = colorname.split(delim)
        if len(parts) == 1:
            return (colorname, 1)
        elif len(parts) == 2:
            return (parts[0], int(parts[1]))
        else:
            raise IndexError(f'{colorname} has too many depths')

    @staticmethod
    def combine_colors(
        colorname: str,
        depth: int,
        delim: str,
    ) -> str:
        return f'{colorname}{delim}{depth}'

    @cached_property
    def valid_colors(self):
        # takes the same black 4 times
        return sum(
            [
                [
                    self.combine_colors(
                        colorname=color,
                        depth=depth,
                        delim=self.depth_delim,
                    ) for color in colors.valid_colors
                ]
                for depth, colors in self.color_sets.items()
            ],
            []
        )

    @cached_property
    def valid_colors_unique(self):
        existing_colors, relevant_names = [], []
        for color_name in self.valid_colors:
            color = self.get_color(color_name)
            if color in existing_colors or 'white' in color_name:
                continue
            existing_colors.append(color)
            relevant_names.append(color_name)
        return relevant_names

    def get_color(self, color: str) -> str:
        color, depth = self.split_color_name(color, self.depth_delim)
        try:
            return self.color_sets[depth].get_color(color)
        except KeyError:
            raise KeyError(f'Wrong shade-depth {depth}, '
            f'must be one of {list(self.color_sets.keys())}')

    def __getitem__(self, key: int):
        return self.color_sets.get(key)



class ColorOfObj(UserDict):
    def __init__(
        self,
        color_set: ColorSet,
        preset_data: dict[Any, str] = {},
        available_colors: list = None, 
    ):
        self.data = preset_data
        self.color_set = color_set
        self.available_colors = available_colors if available_colors \
            else color_set.valid_colors

    @classmethod
    def init_for_single_shade(
        cls,
        depth: int=1,
        fix_colors: dict[Any, str]={},
    ):
        color_set = ColorSet.init(depth)
        missing_colors = list(
            set(fix_colors.values()) 
            - set(color_set.valid_colors)
        )
        if len(missing_colors) > 0:
            raise ValueError(f"{', '.join(missing_colors)} not available")
        available_colors = [
            color for color in color_set.valid_colors 
            if color not in fix_colors.values() and color != 'white'
        ]
        preset_data = {
            key: color_set.get_color(color) 
            for key, color in fix_colors.items()
        }
        return cls(
            preset_data=preset_data,
            color_set=color_set,
            available_colors=available_colors,
        )

    @classmethod
    def init_for_multi_shade(
        cls,
        fix_colors: dict[Any, str]={},
        **set_kwargs,
    ):
        color_set = ColorSetRavelled.init(**set_kwargs)
        missing_colors = list(
            set(fix_colors.values()) 
            - set(color_set.valid_colors_unique)
        )
        if len(missing_colors) > 0:
            raise ValueError(f"{', '.join(missing_colors)} not available")
        available_colors = [
            color for color in color_set.valid_colors_unique
            if color not in fix_colors.values() and color != 'white'
        ]
        preset_data = {
            key: color_set.get_color(color)
            for key, color in fix_colors.items()
        }
        obj = cls(
            preset_data=preset_data,
            color_set=color_set,
            available_colors=available_colors,
        )
        return obj

    @classmethod
    def init(
        cls,
        depth: int | list[int] =None,
        fix_colors: dict[Any, str]={},
    ):
        if depth is None or isinstance(depth, list):
            return cls.init_for_multi_shade(
                fix_colors=fix_colors
        )
        elif isinstance(depth, nbr.Integral):
            return cls.init_for_single_shade(
                depth=depth, 
                fix_colors=fix_colors,
            )
        else:
            raise ValueError(f'{depth} must be an int or a list of ints') 

    def __getitem__(self, key: Any):
        if key not in self.data.keys():
            if not self.available_colors:
                raise ValueError(f'too many items, too few colors')
            color = self.color_set.get_color(self.available_colors[0])
            self.data[key] = color
            self.available_colors.pop(0)
            return color
        else:
            return self.data[key]

    def __call__(self, key: str) -> str:
        return self[key]

    def __getattr__(self, key: str) -> str:
        return self.data.get(key)

    @property
    def registered_objs(self):
        return [str(key) for key in self.data.keys()]