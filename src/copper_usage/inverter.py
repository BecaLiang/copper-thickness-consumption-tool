from abc import ABC, abstractmethod

import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm
from functools import partial, cached_property

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from copper_usage.thickness_calculation import ThicknessCalculation


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ParameterFitResult:
    
    params: np.ndarray
    inv_hessian: np.ndarray

    @cached_property
    def hessian(self):
        return np.linalg.inv(self.inv_hessian)

    def calculate_distance(self, point: np.ndarray):
        # use with caution; the formula is a (valid!) approximation around the minimum
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        assert point.shape == self.params.shape
        V = point - self.params
        return np.sqrt(
            np.dot(
                np.dot(V.T, self.hessian), 
                V,
            )
        )


class ErrorModel(ABC):
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> ParameterFitResult:
        pass


class GaussianErrorModel_(ErrorModel):
    
    def __call__(
            self,
            margin: float,
            min_required: float,
            empirical_sigma: float,
    ):
        return norm.isf(1 - margin, min_required, empirical_sigma)


class GaussianErrorModel(ErrorModel):

    @staticmethod
    def score(predict, xs, y, minreq, sigma_v):
        N = norm.cdf(minreq, predict(xs), sigma_v)
        return (N - y) ** 2

    def __call__(
            self,
            calculator: ThicknessCalculation,
            margin: float,
            min_required: float,
            p0: list[float]=None,
            empirical_sigma: float=None,
    ):

        minim_res = minimize(
            partial(
                self.score, 
                calculator.predict_from_list,
            ), 
            x0=p0 or calculator._X0,
            args=(margin, min_required, empirical_sigma or calculator._y_width)
        )

        # Attention: fit might fail without throwing an exception!
        # Adapting initial values might be necessary in that case
        # TODO: deal with the issue described above
        return ParameterFitResult(
            params=minim_res.x,
            inv_hessian=minim_res.hess_inv,
        )