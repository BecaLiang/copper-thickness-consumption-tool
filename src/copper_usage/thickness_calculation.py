from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass

import pandas as pd
import numpy as np

from typing import Any

from copper_usage.sop_slicer import SOPSlicer
from functools import wraps, partial

from scipy.optimize import curve_fit


@dataclass
class Constraint:

    lower: float | None=None
    upper: float | None=None

    def get(self):
        return [self.lower or -np.inf, self.upper or np.inf]
    
    @classmethod
    def init_dict_from_dict(cls, constraints: dict[str, dict[str, float]]=None):
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

    time_column: str='time_pattern'
    current_density_column: str='current_pattern'
    target_column: str='minimal_thickness'
    thickness_column: str | None=None
    spray_column: str | None=None

    constraints: dict[str, Constraint]=None

    _fitted_parameters: list[str] | None=None

    @classmethod
    def init_from_config(cls, cfg: dict=None, *args, **kwargs):
        
        if cfg is None:
            return cls()
        
        obj = cls(
            time_column = cfg.get('time_column', cls.time_column),
            current_density_column = cfg.get('current_density_column', cls.current_density_column),
            target_column = cfg.get('target_column', cls.target_column),
            thickness_column = cfg.get('thickness_column', cls.thickness_column),
            spray_column = cfg.get('spray_column', cls.spray_column),
            constraints = Constraint.init_dict_from_dict(
                cfg.get('constraints', {})
            ),
        )
        obj.set_fit_parameters(cfg.get('fit_parameters', []))
        
        return obj

    @property
    def all_columns(self) -> list[str]:
        return [
            getattr(self, field) for field in  
            self.__pydantic_fields__.keys() if 'column' in field
        ]

    @property
    def relevant_columns(self) -> list[str]:
        return [column for column in self.all_columns if column is not None]
    
    @property
    def fitted_parameters(self) -> list[str]:
        if self._fitted_parameters is None:
            return [self.time_column, self.current_density_column]
        return self._fitted_parameters
    
    def set_fit_parameters(self, columns: list[str], by_id: bool=True):
        # raises AttributeError
        if by_id:
            self._fitted_parameters = [getattr(self, c) for c in columns]
        else:
            self._fitted_parameters = columns
    
    def get_boundaries(self, columns: list[str]=None) -> list[list[int]]:
    
        if self.constraints is None:
            return None
        
        bounds = []
        for c in columns or self.fitted_parameters:
            bounds.append(self.constraints.get(c, Constraint()))
        
        return [b.get() for b in bounds]


MATHMODELS = {}
THICKNESS_CALCULATIONS = {}


def make_singles_scalar(f):
    @wraps(f)
    def inner_function(*args, **kwargs):
        result = f(*args, **kwargs)
        if np.ndim(result) == 1:
            if len(result) == 1:
                return next(x for x in result)
        return result
    return inner_function


def register_calculator(name: str=None):
    def decorator(cls):
        key = name or cls.__name__.lower()
        THICKNESS_CALCULATIONS[key] = cls
        return cls
    return decorator


def register_mathmodel(name: str=None):
    def decorator(cls):
        key = name or cls.__name__.lower()
        MATHMODELS[key] = cls
        return cls
    return decorator


class MathematicalThicknessModel(ABC):

    default_start_values: list[float]=None

    def __init__(self, column_ids: dict[str, int]=None):
        self.columns = column_ids
    
    @abstractmethod
    def __call__(self, X: np.array, *args):
        pass


@register_mathmodel('plain_linear')
class PlainLinearModel(MathematicalThicknessModel):

    default_start_values = [1, 0]

    def __call__(self, X, slope, offset):
        return slope * X[0] * X[1] + offset


@register_mathmodel('linear_board')
class PlainLinearModelWithBoard(MathematicalThicknessModel):

    default_start_values = [1, 1, 0]

    def __call__(self, X, slope, bthick, offset):
        return slope * X[0] * X[1] + bthick * X[2] + offset


@register_mathmodel('linear_board_and_spray')
class PlainLinearModelWithBoardAndSpray(MathematicalThicknessModel):

    default_start_values = [1, 1, 1, 0]

    def __call__(self, X, slope, bthick, pspray, offset):
        return slope * X[0] * X[1] + bthick * X[2] + pspray * X[3] + offset


class ThicknessCalculation(ABC):
    
    def __init__(
            self, 
            data_columns: DataColumns,
            data_slicer: SOPSlicer,
            calc: MathematicalThicknessModel,
            start_values: list[float],
            # data_names: ,
        ):

        self.data_columns = data_columns

        self.slicer = data_slicer
        self.calc = calc
        self.start_values = start_values

        self.fitted_params = None
        self.fitted_cov = None

        self._y_width = None
        self._X0 = None

    def is_in_charge(self, **kwargs):
        return self.slicer.is_in_charge(**kwargs)

    @abstractmethod
    def _fit(self, df, start_values: list[float], verbose: bool=False):
        pass

    def fit(self, df: pd.DataFrame, start_values: list[float]=None, verbose: bool=False):
        dfprep = self.clean_data(df, verbose=verbose)
        self._fit(dfprep, start_values=start_values, verbose=verbose)

    @abstractmethod
    def _predict(self) -> pd.Series | float:
        pass

    def clean_data(self, df: pd.DataFrame, verbose: float=False):
        dfproc = self.slicer.slice_data(df)
        dfnotNull = dfproc[
            ~dfproc[self.data_columns.relevant_columns]
            .isnull()
            .max(axis=1)
        ]
        if dfnotNull.shape[0] != dfproc.shape[0] and verbose:
            print(self.slicer, '\tNot-Null Entries:', dfnotNull.shape[0], 'of', dfproc.shape[0])

        assert len(dfnotNull) > 0
        return dfnotNull
    
    def predict(self, df: pd.DataFrame=None, **kwargs):
        if df is None:
            return self._predict(None, **kwargs)
        else:
            return self.bulk_predict(df=df, **kwargs)

    def bulk_predict(
            self, 
            df: pd.DataFrame, 
            name: str='thickness_prediction',
            valid_only: bool=False, 
            return_series: bool=True,
        ):
        dfprep = self.clean_data(df)
        arr_pred = self._predict(dfprep)
        if valid_only:
            if return_series:
                return pd.Series(
                    index=dfprep.index,
                    data=arr_pred,
                    name=name,
                )
            else:
                return arr_pred
        else:
            ps = pd.Series(
                    index=dfprep.index,
                    data=arr_pred,
                    name=name,
                ).reindex(self.slicer.slice_data(df).index)
            if return_series:
                return ps
            else:
                return ps.values

    @make_singles_scalar
    def calculate(self, df: pd.DataFrame):
        dfproc = self.slicer.slice_data(df)
        # TODO assert not empty
        return self.predict(dfproc)
    
    def build_predict_from_list(self, **kwargs):
        return self.predict_from_list
    
    def extract_fixed_values(self, **kwargs) -> dict[int, Any]:
        return {}


@register_calculator('plain_linear_calculation')
class PlainLinearThicknessCalculation(ThicknessCalculation):

    def _pd_to_numpy(self, df: pd.DataFrame) -> np.array:
        return df[self.data_columns.fitted_parameters].values.T

    def _fit(
            self, 
            df: pd.DataFrame,
            start_values: list[float]=None,
            verbose: bool=False,
        ):

        Xin = self._pd_to_numpy(df)
        Ytrain = df[self.data_columns.target_column].values

        self.fitted_params, self.fitted_cov = curve_fit(
            self.calc, 
            Xin,
            Ytrain,
            p0=self.start_values if start_values is None else start_values,
        )

        # TODO: Bit of a tricky business; might be not precise enough for an estimtor
        # (dependent on N-input)
        self._y_width = np.std(self.predict_from_list(Xin) - Ytrain, ddof=1)
        self._X0 = np.mean(Xin, axis=1)

    def _predict(self, df=None, *args, **kwargs):
        assert self.fitted_params is not None
        if df is not None:
            return self.calc(self._pd_to_numpy(df), *self.fitted_params)
        else:
            return self.calc(self._kwargs_to_numpy(**kwargs), *self.fitted_params)
        
    def predict_from_list(self, X):
        return self.calc(X, *self.fitted_params)

    # predict_for_optimization = partialmethod(predict_from_list)

    def _kwargs_to_numpy(self, **kwargs):
        return np.array([kwargs[lc] for lc in self.simple_linear_column])
    
    def get_uncertainty_of_x(self, X):
        pass


@register_calculator('board_linear_calculation')
class BoardInclusiveLinearThicknessCalculation(PlainLinearThicknessCalculation):

<<<<<<< HEAD
    def build_predict_from_list(self, fixes: dict=None, **kwargs) -> list[dict[int, Any]]:
=======
    fit_parameters = ['time_column', 'current_density_column', 'thickness_column']

    def build_predict_from_list(self, fixes: dict=None, **kwargs):
>>>>>>> include_spray

        for to_fix, fix in (fixes or {}).items():
            self._X0[to_fix] = fix
        
        def new_predict_from_list(X):
            for to_fix, fix in (fixes or {}).items():
                X[to_fix] = fix
            return self.predict_from_list(X)
        
        return new_predict_from_list
    
    def extract_fixed_values(self, **kwargs) -> dict[int, Any]:
        
        fix_position = self.data_columns.fitted_parameters.index(
            self.data_columns.thickness_column
        )
        
        try:
            return {
                fix_position: kwargs['thickness_column']
            }
        except KeyError:
            return {}