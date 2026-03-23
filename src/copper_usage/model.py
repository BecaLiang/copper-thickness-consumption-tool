from copper_usage.thickness_calculation import ThicknessCalculation

from copper_usage.inverter import (
    ErrorModel,
    GaussianErrorModel,
    ParameterFitResult,
)
from copper_usage.feature_containers import (
    MachineFeatureContainer, 
    BoardFeatureContainer,
)

import pandas as pd


class Model:

    def __init__(
            self, 
            thickness_calculations: list[ThicknessCalculation],
            error_model: ErrorModel=None,
            default_margin: float=None,
        ):
        self.thickness_calculations = thickness_calculations
        self.error_model = error_model or GaussianErrorModel()
        self.default_margin = default_margin

    def _assert_unambiguous_calculation(self):
         # assert self.
         pass

    def _select_calculator_idx(self, *args, **kwargs):
        for idx, calc in enumerate(self.thickness_calculations):
             if calc.is_in_charge(*args, **kwargs):
                  return idx
        return None

    def fit(self, df: pd.DataFrame, verbose: bool=False):
        for tc in self.thickness_calculations:
            tc.fit(df, verbose=verbose)

    def predict(self, board: BoardFeatureContainer):
        return self.predict_single_board(
            minimal_thickness=BoardFeatureContainer.minimal_thickness,
            margin=BoardFeatureContainer.margin,
            is_vcp=BoardFeatureContainer.is_vcp,
            Ratio=BoardFeatureContainer.Ratio,
            thickness=BoardFeatureContainer.thickness,
        )

    def predict_single_board(
            self, 
            minimal_thickness: float,
            margin: float=None,
            p0: list[float]=None,
            sigma: float=None,
            **kwargs,
        ) -> ParameterFitResult:
            cid = self._select_calculator_idx(**kwargs)
            assert cid is not None
            calc = self.thickness_calculations[cid]
            fixes = calc.extract_fixed_values(**kwargs)
            fitted_parameters = self.error_model(
                calc, 
                margin=margin or self.default_margin,
                min_required=minimal_thickness,
                p0=p0,
                empirical_sigma=sigma,
                fixes=fixes,
            )
            pass