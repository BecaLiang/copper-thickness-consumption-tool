from abc import ABC, abstractmethod

from scipy.optimize import curve_fit

import pandas as pd
import numpy as np

from typing import Any
from numpy.typing import NDArray
from functools import wraps

import warnings
import logging


from copper_usage.data_columns import DataColumns
from copper_usage.sop_slicer import SOPSlicer


logging.basicConfig(
    filename="thickness_calculation_run.log",
    level=logging.WARNING,
    filemode='w',
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.captureWarnings(True)


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

    default_start_values = [36, 1, 1, 0]

    def __call__(self, X, slope, bthick, pspray, offset):
        return slope * X[0] * X[1] + bthick * X[2] + pspray * X[3] + offset
    

@register_mathmodel('linear_board_spray_fixed_time') 
class LinearModelWithSprayFixedTime(MathematicalThicknessModel):

    default_start_values = [1, 1, 1, 0]

    def __init__(
            self, 
            fixed_time: float,
            column_ids = None,
        ):
        super().__init__(column_ids)
        self._fixed_time = fixed_time

    def __call__(self, X, slope, bthick, pspray, offset):
        return slope * X[0] * self._fixed_time + bthick * X[1] + pspray * X[2] + offset


class ThicknessCalculation(ABC):
    
    def __init__(
            self, 
            data_columns: DataColumns,
            data_slicer: SOPSlicer,
            calc: MathematicalThicknessModel,
            start_values: list[float],
            epsilon: float=1e-3,
            board_specifics: list[str]=None,
        ):

        self.data_columns = data_columns

        self.slicer = data_slicer
        self.calc = calc
        self.start_values = start_values

        self.fitted_params = None
        self.fitted_cov = None

        self._y_width = None
        self._X0 = None

        self._epsilon = epsilon

        if board_specifics is None:
            self.board_specifics = []
        else:
            assert len(set(board_specifics) - set(self.data_columns.fitted_parameters)) == 0
            self.board_specifics = board_specifics

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

        for col, v in self.data_columns.fixed_values.items():
            if np.any(np.abs(dfnotNull[self.data_columns[col]] - v) > self._epsilon):
                ps_valid = np.abs(dfnotNull[self.data_columns[col]] - v) < self._epsilon
                Ninvalid, Nvalid = (~ps_valid).sum(), ps_valid.sum()
                warnings.warn(
                    f'\nATTENTION: {Ninvalid} values deviate too far from {v} for {col} - '
                    f'that amounts to {Ninvalid / (Ninvalid + Nvalid) * 100:.1f}%. '
                    f'Removing {Ninvalid} from {Ninvalid + Nvalid} entries'
                )
                dfnotNull = dfnotNull[ps_valid]

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
    
    @staticmethod
    def apply_fixes(
        X: list[float] | NDArray[np.float64],
        fixes: dict[int, float]=None
    ) -> list | NDArray[np.float64]:
        for to_fix, fix in (fixes or {}).items():
            X[to_fix] = fix
        return X


@register_calculator('plain_linear_calculation')
class PlainLinearThicknessCalculation(ThicknessCalculation):

    def _pd_to_numpy(self, df: pd.DataFrame) -> np.array:
        return df[self.data_columns.fit_columns].values.T

    def _fit(
            self, 
            df: pd.DataFrame,
            start_values: list[float]=None,
            verbose: bool=False,
        ):

        Xin = self._pd_to_numpy(df)
        Ytrain = df[self.data_columns.target_column].values

        if start_values is None:
            if self.start_values is not None:
                assert len(self.start_values) == len(self.calc.default_start_values)
                start_values = self.start_values
            else:
                start_values = self._X0

        self.fitted_params, self.fitted_cov = curve_fit(
            self.calc, 
            Xin,
            Ytrain,
            # p0=self.start_values if start_values is None else start_values,
            p0=start_values,
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

    def _kwargs_to_numpy(self, **kwargs):
        return np.array([kwargs[lc] for lc in self.simple_linear_column])
    
    def get_uncertainty_of_x(self, X):
        pass


@register_calculator('board_linear_calculation')
class BoardInclusiveLinearThicknessCalculation(PlainLinearThicknessCalculation):

    def __init__(
            self,
            data_columns, 
            data_slicer,
            calc,
            start_values,
            epsilon=0.001, 
            board_specifics=None,
        ):
        if board_specifics is None:
            _board_specifics = ['board_thickness']
        else:
            _board_specifics = board_specifics
        if not isinstance(_board_specifics, list):
            raise TypeError

        super().__init__(
            data_columns,
            data_slicer,
            calc,
            start_values,
            epsilon=epsilon,
            board_specifics=_board_specifics,
        )

    def build_predict_from_list(
            self, fixes: 
            dict=None, 
            **kwargs
        ) -> list[dict[int, Any]]:
    
        for to_fix, fix in (fixes or {}).items():
            self._X0[to_fix] = fix
        
        def new_predict_from_list(X):
            X = ThicknessCalculation.apply_fixes(X, fixes)
            return self.predict_from_list(X)
        
        return new_predict_from_list

    def extract_fixed_values(self, fix_columns: list[str]=None, **kwargs) -> dict[int, Any]:

        assert isinstance(fix_columns, list)

        if fix_columns is None:
            return {}

        fix_positions = {
            self.data_columns.fitted_parameters.index(
                fc
            ): kwargs[fc]
            for fc in fix_columns
        }

        return fix_positions