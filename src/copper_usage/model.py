from copper_usage.thickness_calculation import ThicknessCalculation

from copper_usage.inverter import (
    ErrorModel,
    GaussianErrorModel,
    ParameterFitResult,
)
from copper_usage.feature_containers import BoardFeatureContainer

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
         pass

    def get_data_column(
                self,
                is_vcp: bool,
                Ratio: float=None,
                **kwargs,
        ):
            if Ratio is None:
                tcs = [tc for tc in self.thickness_calculations if tc.slicer.is_vcp == is_vcp]
                if len(tcs) == 0:
                    raise IndexError
                else:
                    return tcs[0].data_columns
            else:
                return self.thickness_calculations[
                    self._select_calculator_idx(
                        is_vcp=is_vcp,
                        Ratio=Ratio,
                    )
                ].data_columns

    def _select_calculator_idx(
              self,
              raise_if_missing: bool=True,
              *args,
              **kwargs,
        ):
        for idx, calc in enumerate(self.thickness_calculations):
             if calc.is_in_charge(*args, **kwargs):
                  return idx
        if raise_if_missing:
             raise KeyError
        else:
            return None

    def fit(self, df: pd.DataFrame, verbose: bool=False):
        for tc in self.thickness_calculations:
            tc.fit(df, verbose=verbose)

    def predict(self, board: BoardFeatureContainer):
        return self.predict_single_board(
            minimal_thickness=board.minimal_thickness,
            margin=board.margin,
            is_vcp=board.is_vcp,
            Ratio=board.Ratio,
            board_thickness=board.board_thickness,
        )

    def predict_single_board(
            self, 
            minimal_thickness: float,
            margin: float=None,
            p0: list[float]=None,
            sigma: float=None,
            fix_columns: list[str]=['board_thickness'],
            **kwargs,
        ) -> ParameterFitResult:
            cid = self._select_calculator_idx(**kwargs)
            calc = self.thickness_calculations[cid]
            fixes = calc.extract_fixed_values(fix_columns, **kwargs)
            fitted_parameters = self.error_model(
                calc, 
                margin=margin or self.default_margin,
                min_required=minimal_thickness,
                p0=p0,
                empirical_sigma=sigma,
                fixes=fixes,
            )
            result = calc.data_columns.spawn_board_features(
                fitted_values=fitted_parameters.params,
                fixes=fixes,
            )
            return result

    def get_mandatory_not_null(self) -> list[str]:
        cols = []
        for tcalc in self.thickness_calculations:
             cols += tcalc.slicer.get_mandatory_not_null()
        return list(set(cols))