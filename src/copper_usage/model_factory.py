import yaml

from pathlib import Path

from copper_usage.data_columns import DataColumns
from copper_usage.thickness_calculation import (
    MATHMODELS,
    THICKNESS_CALCULATIONS,
    PlainLinearThicknessCalculation,
)
from copper_usage.inverter import GaussianErrorModel
from copper_usage.sop_slicer import VCPLineRatioSlicer

from copper_usage.model import Model


class ModelFactory:

    @staticmethod
    def build_single_calculation_slice(
        slicer: VCPLineRatioSlicer,
        cfg: dict,
    ):
        
        columns = DataColumns.init_from_config(
            cfg.get('data_columns', None),
        )

        math_calc = MATHMODELS[
            cfg.get('fit_model', 'plain_linear')
        ](
            **cfg.get('fit_model_kwargs', {}),
        )

        return THICKNESS_CALCULATIONS[
            cfg.get('calculation_model', PlainLinearThicknessCalculation)
        ](
            data_columns = columns,
            data_slicer = slicer,
            calc = math_calc,
            start_values = cfg.get(
                'start_values', 
                math_calc.default_start_values
            ),
        )

    @staticmethod
    def init_separate_vcp(config: dict, *args, **kwargs) -> Model:
        
        slicers = VCPLineRatioSlicer.initialize_slices(
            config['slices'],
        )

        vcp_cfg = config['vcp']
        # common = config['common']
        non_vcp_cfg = config['non_vcp']

        calculations = [
            ModelFactory.build_single_calculation_slice(
                slicer=slicer,
                cfg=vcp_cfg if slicer.is_vcp else non_vcp_cfg,
            ) for slicer in slicers
        ]

        # TODO: Make configurable
        return Model(
            calculations,
            error_model=GaussianErrorModel()
        )
    
    @staticmethod
    def build_model_from_config(cfg_file: str='default_models.yaml'):
        if Path(cfg_file).is_file():
            with open(Path(cfg_file)) as yin:
                return ModelFactory.init_separate_vcp(yaml.safe_load(yin))
        else:
            path = Path(__file__).resolve().parent / Path('config') / cfg_file
            with open(path) as yin:
                return ModelFactory.init_separate_vcp(yaml.safe_load(yin))