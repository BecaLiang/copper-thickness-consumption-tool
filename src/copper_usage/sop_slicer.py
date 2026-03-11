from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass
from pydantic import NonNegativeFloat

import numpy as np
import pandas as pd


SLICER_REGISTRY = {}


def register_slicerclass(name: str=None):
    def decorator(cls):
        key = name or cls.__name__.lower()
        SLICER_REGISTRY[key] = cls
        return cls
    return decorator


class SOPSlicer(ABC):
    
    @abstractmethod
    def slice_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def initialize_slices(cls, config: dict | list, *args, **kwargs):
        pass

    @abstractmethod
    def is_in_charge(self, **kwargs):
        pass


@register_slicerclass('vcp_and_ratio')
@dataclass
class VCPLineRatioSlicer(SOPSlicer):

    is_vcp: bool
    ratio_min: NonNegativeFloat
    ratio_max: NonNegativeFloat

    vcp_column: str='is_vcp'
    ratio_column: str='Ratio'

    def __post_init__(self):
        assert self.ratio_min < self.ratio_max

    def __lt__(self, rhs):
        if self.is_vcp == rhs.is_vcp:
            return self.ratio_min < rhs.ratio_min
        else:
            return rhs.is_vcp
        
    def __str__(self):
        vcp_info = 'NON-VCP' if not self.is_vcp else 'VCP'
        return f'{vcp_info} / {self.ratio_column}: [{self.ratio_min:.1f}, {self.ratio_max:.1f})'

    def slice_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_vcp:
            return df[
                df[self.vcp_column]
                & (df[self.ratio_column] >= self.ratio_min)
                & (df[self.ratio_column] < self.ratio_max)
            ].copy()
        else:
            return df[
                ~df[self.vcp_column]
                & (df[self.ratio_column] >= self.ratio_min)
                & (df[self.ratio_column] < self.ratio_max)
            ].copy()
        
    def is_in_charge(self, **kwargs):
        is_vcp = kwargs.get(self.vcp_column, None)
        ratio_value = kwargs.get(self.ratio_column, None)
        assert is_vcp is not None and ratio_value is not None
        return self.contains(ratio_value)

    def contains(self, value: float):
        return (self.ratio_min >= value) & (value < self.ratio_min)

    @classmethod    
    def initialize_slices(cls, config: dict):

        slice_boarders = config.get('slice_boarders', None)
        overflow_slice = config.get('overflow', False)
        underflow_slice = config.get('underflow', False)

        if isinstance(slice_boarders, (list, tuple)):
            slice_boarders = sorted(slice_boarders)
            slices = sum(
                [
                    [
                        cls(is_vcp=True, ratio_min=cmin, ratio_max=cmax),
                        cls(is_vcp=False, ratio_min=cmin, ratio_max=cmax),
                    ] for cmin, cmax in zip(slice_boarders[:-1], slice_boarders[1:])
                ],
                []
            )
            
            if overflow_slice:
                slices.append(cls(is_vcp=False, ratio_min=slice_boarders[-1], ratio_max=np.inf))
                slices.append(cls(is_vcp=True, ratio_min=slice_boarders[-1], ratio_max=np.inf))
            if underflow_slice and min(slice_boarders) > 0:
                slices.append(cls(is_vcp=False, ratio_min=0, ratio_max=slice_boarders[0]))
                slices.append(cls(is_vcp=True, ratio_min=0, ratio_max=slice_boarders[0]))
        
        elif isinstance(slice_boarders, dict):
            vcp_boarders = sorted(slice_boarders['vcp'])
            non_vcp_boarders = sorted(slice_boarders['non_vcp'])
            assert isinstance(vcp_boarders, (list, tuple)) and isinstance(non_vcp_boarders, (list, tuple))
            slices = [
                cls(is_vcp=False, ratio_min=cmin, ratio_max=cmax)
                for cmin, cmax in zip(non_vcp_boarders[:-1], non_vcp_boarders[1:])
            ] + [
                cls(is_vcp=True, ratio_min=cmin, ratio_max=cmax)
                for cmin, cmax in zip(vcp_boarders[:-1], vcp_boarders[1:])
            ]
            
            if overflow_slice:
                slices.append(cls(is_vcp=False, ratio_min=non_vcp_boarders[-1], ratio_max=np.inf))
                slices.append(cls(is_vcp=True, ratio_min=vcp_boarders[-1], ratio_max=np.inf))
            if underflow_slice:
                if min(non_vcp_boarders) > 0:
                    slices.append(cls(is_vcp=False, ratio_min=0, ratio_max=non_vcp_boarders[0]))
                if min(vcp_boarders) > 0:
                    slices.append(cls(is_vcp=True, ratio_min=0, ratio_max=vcp_boarders[0]))
        
        else:
            raise TypeError
        
        return sorted(slices)
    

class FullDataSlice(SOPSlicer):
    
    def slice_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    

class SlicerFactory:
    
    def spawn_slices(self, name: str, config: dict | list) -> list[SOPSlicer]:
    
        if name not in SLICER_REGISTRY.keys():
            raise KeyError
        
        return SLICER_REGISTRY[name].initialize_slices(config)
