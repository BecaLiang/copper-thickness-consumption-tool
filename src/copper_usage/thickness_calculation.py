from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass

import pandas as pd
import numpy as np

from copper_usage.sop_slicer import SOPSlicer
from functools import wraps

from scipy.stats import norm
from scipy.optimize import curve_fit, minimize


@dataclass
class DataColumns:
    time_column: str='time_pattern'
    current_density_column: str='current_pattern'
    target_column: str='minimal_thickness'
    spray_column: str=None

    @property
    def all_columns(self):
        return [getattr(self, field) for field in  self.__pydantic_fields__.keys()]

    @property
    def relevant_columns(self):
        return [column for column in self.all_columns if column is not None]
    
    @property
    def simple_linear_columns(self):
        return [self.time_column, self.current_density_column]


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


class MathematicalThicknessModel(ABC):
    
    @abstractmethod
    def __call__(self, X: np.array, *args):
        pass


class PlainLinearModel(MathematicalThicknessModel):

    def __call__(self, X, slope, offset):
        return slope * X[0] * X[1] + offset


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


class PlainLinearThicknessCalculation(ThicknessCalculation):

    def _pd_to_numpy(self, df: pd.DataFrame) -> np.array:
        return df[self.data_columns.simple_linear_columns].values.T

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