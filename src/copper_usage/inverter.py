from abc import ABC, abstractmethod

import numpy as np

from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

from functools import partial

# from pydantic.dataclasses import dataclass
from typing import Callable

from copper_usage.thickness_calculation import ThicknessCalculation
from copper_usage.feature_containers import ParameterFitResult


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


class Score(ABC):
    pass


class PlainSquareScore(Score):

    def __call__(
            self, 
            predict: Callable, 
            xs: np.ndarray, 
            y: float, 
            minreq: float, 
            sigma_v: float,
        ) -> float:

        N = norm.cdf(
            x=minreq, 
            loc=predict(xs), 
            scale=sigma_v,
        )

        return (N - y) ** 2


class RegularizedScore(Score):

    def __init__(self, reg_lambda: float=1):
        self.reg_lambda = reg_lambda

    def __call__(
            self, 
            predict: Callable, 
            xs: np.ndarray, 
            y: float, 
            minreq: float, 
            sigma_v: float,
        ) -> float:
        N = norm.cdf(
            x=minreq, 
            loc=predict(xs), 
            scale=sigma_v,
        )

        reg_const = np.sqrt(
            np.sum(
                xs * xs
            )
        )

        return (N - y) ** 2 + self.reg_lambda * reg_const


class GaussianErrorModel(ErrorModel):

    def __init__(self, minimizer_kwargs=None):
        self.minimizer_kwargs = minimizer_kwargs or {}

    def __call__(
            self,
            calculator: ThicknessCalculation,
            margin: float,
            min_required: float,
            p0: list[float]=None,
            empirical_sigma: float=None,
            fixes: dict=None,
    ) -> ParameterFitResult:

        score = partial(
            PlainSquareScore(),
            calculator.build_predict_from_list(
                fixes=fixes or {},
                params=calculator.fitted_params,
            ),
        )

        x0 = ThicknessCalculation.apply_fixes(calculator._X0, fixes)
        cons = {
            "type": "eq",
            "fun": score,
            "args": (
                margin,
                min_required,
                empirical_sigma or calculator._y_width,
            ),
        }

        minim_res = minimize(
            lambda x: np.sum((x - calculator._X0) ** 2),
            x0=x0,
            constraints=[cons],
            bounds=calculator.data_columns.get_boundaries(),
        )

        try:
            if isinstance(minim_res.hess_inv, LbfgsInvHessProduct):
                inv_hessian = minim_res.hess_inv.todense()
            else:
                inv_hessian = minim_res.hess_inv
        except (AttributeError, KeyError):
            inv_hessian = np.diag([1] * len(minim_res.x))

        return ParameterFitResult(
            fitted_thickness=calculator.build_predict_from_list(fixes=fixes or {})(minim_res.x),
            params=minim_res.x,
            inv_hessian=inv_hessian,
            iterations=minim_res.nit,
        )
    

class ErrorModelFactory:
    pass