from abc import ABC, abstractmethod

import numpy as np

from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

from functools import partial

from pydantic.dataclasses import dataclass

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


class MainScore(Score):

    def __call__(self, predict, xs, y, minreq, sigma_v) -> float:
        N = norm.cdf(
            x=minreq, 
            loc=predict(xs), 
            scale=sigma_v,
        )

        return (N - y) ** 2


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
            MainScore(),
            calculator.build_predict_from_list(
                fixes=fixes or {}
            ),
        )

        minim_diff = differential_evolution(
            score,
            x0=np.ones_like(calculator._X0) if p0 is None else p0,
            args=(margin, min_required, empirical_sigma or calculator._y_width),
            bounds=calculator.data_columns.get_boundaries(),
            popsize=40,
            seed=42,
        )

        minim_res = minimize(
            score,
            x0=ThicknessCalculation.apply_fixes(minim_diff.x, fixes),
            args=(margin, min_required, empirical_sigma or calculator._y_width),
            bounds=calculator.data_columns.get_boundaries(),
        )

        if isinstance(minim_res.hess_inv, LbfgsInvHessProduct):
            inv_hessian = minim_res.hess_inv.todense()
        else:
            inv_hessian = minim_res.hess_inv
    
        return ParameterFitResult(
            fitted_thickness=calculator.build_predict_from_list(fixes=fixes or {})(minim_res.x),
            params=minim_res.x,
            inv_hessian=inv_hessian,
        )