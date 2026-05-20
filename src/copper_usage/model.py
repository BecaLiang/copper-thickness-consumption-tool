from copper_usage.thickness_calculation import ThicknessCalculation
from copper_usage.data_columns import FitValueCollection

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

    def build_combined_column_dict(self) -> dict[str, str]:
        fvc = FitValueCollection([])
        ratio_column = self.thickness_calculations[0].slicer.ratio_column
        vcp_column = self.thickness_calculations[0].slicer.vcp_column
        for tc in self.thickness_calculations:
            fvc += tc.data_columns.fit_values
            assert ratio_column == tc.slicer.ratio_column
            assert vcp_column == tc.slicer.vcp_column
        return fvc._value_column_map.copy() | {
            'ratio_column': ratio_column,
            'vcp_column': vcp_column,
        }
    
    def extract_board_feature_list(
            self, 
            df: pd.DataFrame, 
            margin: float | str
        ) -> list[BoardFeatureContainer]:
        cols = self.build_combined_column_dict()
        return [
            BoardFeatureContainer(
                margin=margin,
                required_thickness=row[cols['requirement_column']],
                is_vcp=row[cols['vcp_column']],
                Ratio=row[cols['ratio_column']],
                board_thickness=row[cols['board_thickness']],
            ) for _, row in df.iterrows()
        ]


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
            required_thickness=board.required_thickness,
            margin=board.margin,
            is_vcp=board.is_vcp,
            Ratio=board.Ratio,
            board_thickness=board.board_thickness,
        )
    
    def bulk_predict(self, df: pd.DataFrame):
        pass

    def predict_single_board(
            self, 
            required_thickness: float,
            margin: float=None,
            p0: list[float]=None,
            sigma: float=None,
            **kwargs,
        ) -> ParameterFitResult:
            cid = self._select_calculator_idx(**kwargs)
            relevant_calcer = self.thickness_calculations[cid]
            fixes = relevant_calcer.extract_fixed_values(
                 relevant_calcer.board_specifics + relevant_calcer.data_columns.fixed_columns,
                 **(kwargs | relevant_calcer.data_columns.fixed_values),
            )

            actual_start_values = ThicknessCalculation.apply_fixes(
                relevant_calcer.start_values if p0 is None else p0, 
                fixes,
            )

            fitted_parameters = self.error_model(
                relevant_calcer, 
                margin=margin or self.default_margin,
                min_required=required_thickness,
                p0=actual_start_values,
                empirical_sigma=sigma,
                fixes=fixes,
            )

            result = relevant_calcer.data_columns.spawn_board_features(
                fitted_values=fitted_parameters.params,
                fixes=fixes,
            )
            return result

    def get_mandatory_not_null(self) -> list[str]:
        cols = []
        for tcalc in self.thickness_calculations:
             cols += tcalc.slicer.get_mandatory_not_null()
        return list(set(cols))