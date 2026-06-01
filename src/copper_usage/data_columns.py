from __future__ import annotations

import numpy as np

import yaml
import copy

from pathlib import Path

from pydantic.dataclasses import dataclass

from copper_usage.feature_containers import MachineFeatureContainer
from copper_usage.utils import deep_merge_two_dicts
from copper_usage.feature_containers import ParameterFitResult


@dataclass
class FitValue:
    name: str
    column: str | None = None


@dataclass
class FitValueCollection:

    fit_values: list[FitValue]
    _fit_parameters: list[str] | None = None
    _value_column_map: dict[str, str] | None = None

    @staticmethod
    def default_fit_parameters() -> list[str]:
        return [
            'plating_time',
            'current_density',
            'board_thickness',
            'spray_frequency',
            'target',
        ]

    @classmethod
    def spawn_from_defaults(cls):
        return cls(
            [
                FitValue(fp) for fp in cls.default_fit_parameters()
            ]
        )

    @classmethod
    def spawn_from_dict(cls, cfg):
        return cls(
            [
                FitValue(key, column=value) for key, value in cfg.items()
            ]
        )

    def __post_init__(self):
        self._value_column_map = {
            fv.name: fv.column for fv in self.fit_values
        }
        if self._fit_parameters is not None:
            assert all(fp in self._value_column_map for fp in self._fit_parameters)

    def __add__(self, rhs: FitValueCollection):

        common_keys = list(
            set(self._value_column_map.keys())
            & set(rhs._value_column_map.keys())
        )
        if any(
            [
                self._value_column_map[k] != rhs._value_column_map[k]
                for k in common_keys
            ]
        ):
            raise ValueError('Inherent value')
        missing_keys = list(
            set(rhs._value_column_map.keys())
            - set(self._value_column_map.keys())
        )

        new_values = copy.deepcopy(self.fit_values)
        
        for k in missing_keys:
            new_values.append(
                FitValue(
                    name=k,
                    column=rhs[k],
                )
            )
        return FitValueCollection(new_values)

    def __contains__(self, item: str | FitValue):
        if isinstance(item, str):
            return item in [fv.name for fv in self.fit_values]
        elif isinstance(item, FitValue):
            if item.name not in [fv.name for fv in self.fit_values]:
                return False
            return self._value_column_map[item.name] == item.column
        else:
            raise TypeError

    def __getitem__(self, item: str | FitValue):
        if isinstance(item, str):
            return self._value_column_map[item]
        elif isinstance(item, FitValue):
            return self._value_column_map[item.name]
        else:
            raise TypeError
        
    def __len__(self):
        return len(self.fit_values)

    def get_column(self, name: str, raise_if_missing: bool=False):
        if raise_if_missing:
            return self._value_column_map[name]
        else:
            return self._value_column_map.get(name)

    @property
    def fit_parameters(self):
        if self._fit_parameters is None:
            self._fit_parameters = [fv.name for fv in self.fit_values]
        return self._fit_parameters

    def columns(self, only_fit: bool=False) -> list[str]:
        if only_fit:
            return [self.get_column(fp) for fp in self.fit_parameters]
        else:
            return [fv.column for fv in self.fit_values]


@dataclass
class Constraint:

    lower: float | None=None
    upper: float | None=None

    def get(self):
        return [
            -np.inf if self.lower is None else self.lower,
            np.inf if self.upper is None else self.upper,
        ]

    @classmethod
    def init_dict_from_dict(
        cls,
        constraints: dict[str, dict[str, float]]=None
    ):
        if constraints is None:
            return None
        obj_dict = {}
        for column, constrs in constraints.items():
            if len(constrs) == 0:
                raise ValueError
            obj_dict[column] = cls(
                lower=constrs.get('lower'),
                upper=constrs.get('upper'),
            )
        return obj_dict


@dataclass
class DataColumns:

    # Notiz aus der Zukunft, ist das wirklich noetig, 
    # koennte man sich nicht eine von beiden Klassen sparen,
    # indem man beide zusammenfuehrt?

    # Sollte vllt. sogar ein Singleton sein. Eine Ausgabe für alle.

    fit_values: FitValueCollection | None=None
    constraints: dict[str, Constraint]=None
    fixed_values: dict[str, float]=None
    _fitted_parameters: list[str] | None=None
    # _fit_values: FitValueCollection | None=None

    def __post_init__(self):
        if self.fit_values is None:
            self.fit_values = FitValueCollection.spawn_from_defaults()

    @classmethod
    def init_from_file(
            cls,
            cfg_file_name: str='default_models.yaml',
            cfg_key: str='vcp',
    ):
        cfg_file = Path(cfg_file_name)
        if not cfg_file.is_file():
            cfg_file = Path(__file__).parent / 'src' / 'copper_usage' / 'config' /  cfg_file
        with open(cfg_file) as yin:
            cfg = yaml.safe_load(yin)
        ind_cfg = deep_merge_two_dicts(
            cfg.get('common_data_columns', {}),
            cfg[cfg_key]['data_columns'],
        )

        return cls.init_from_config(ind_cfg)

    @classmethod
    def init_from_config(
            cls,
            cfg: dict,
            confirm_target: bool=True,
            use_default: bool=False,
            *args,
            **kwargs
        ):

        if use_default:
            fullinfo = {
                fp: cfg.get(fp) for fp in FitValueCollection.default_fit_parameters()
            }
        else:
            fullinfo = {
                k: v for k, v in cfg.items() if isinstance(v, str)
            }
        fvc = FitValueCollection.spawn_from_dict(
            {k: v for k, v in fullinfo.items() if v is not None}
        )
        obj = cls(
            fvc,
            cfg.get('constraints', {}),
            fixed_values=cfg.get('fixed_values', {}),
        )
        obj.set_fit_parameters(cfg.get('fit_parameters', []))

        if confirm_target:
            assert obj.target_column is not None

        return obj

    def spawn_board_features(
            self,
            # fitted_values: list[float],
            fit_result: ParameterFitResult,
            fixes: dict=None,
            **kwargs,
        ) -> MachineFeatureContainer:

        bfc_kwargs = {
            p: v for p, v in zip(
                self.fitted_parameters, 
                fit_result.params,
            )
        }

        return MachineFeatureContainer(
            **bfc_kwargs, 
            target_thickness=fit_result.fitted_thickness,
        )
    
    def __getitem__(self, key):
        return self.fit_values.get_column(key)

    @property
    def all_columns(self) -> list[str]:
        return self.fit_values.columns()

    @property
    def relevant_columns(self) -> list[str]:
        return self.fit_columns + [
            self.fit_values.get_column('target', raise_if_missing=True)
        ]

    @property
    def fitted_parameters(self) -> list[str]:
        if self._fitted_parameters is None:
            return [
                'plating_time',
                'current_density',
            ]
        return self._fitted_parameters

    @property
    def fit_columns(self) -> list[str]:
        return [
            self.fit_values.get_column(fp) for fp in self.fitted_parameters
        ]

    @property
    def target_column(self) -> str:
        return self.fit_values.get_column('target', raise_if_missing=True)
    
    @property
    def fixed_columns(self) -> list[str]:
        return list(self.fixed_values.keys())

    def set_fit_parameters(self, fit_value_names: list[str]):
        self._fitted_parameters = [
            fname for fname in fit_value_names if fname in self.fit_values
        ]

    def get_boundaries(
            self, 
            columns: list[str]=None, 
            is_finite: bool=False
        ) -> list[list[int]]:

        if self.constraints is None:
            return None
        bounds = []
    
        for c in columns or self.fitted_parameters:
            bounds.append(self.constraints.get(c, Constraint()))
        return [b.get() for b in bounds]