import numpy as np

from pydantic.dataclasses import dataclass
from pydantic import (
    PositiveFloat, 
    confloat, 
)
from pydantic import ConfigDict

from functools import cached_property

@dataclass
class BoardFeatureContainer:
    margin: confloat(gt=0, lt=1)
    required_thickness: PositiveFloat
    is_vcp: bool
    Ratio: PositiveFloat
    board_thickness: PositiveFloat


@dataclass
class MachineFeatureContainer:

    plating_time: PositiveFloat
    current_density: PositiveFloat
    spray_frequency: PositiveFloat | None = None
    target_thickness: PositiveFloat | None = None

    _meta: dict | None = None

    def __str__(self) -> str:
        ptstr = f"plating-time: {self.plating_time} min"
        cdstr = f"current-density: {self.current_density} A/cm^2"
        if self.spray_frequency is None:
            return ptstr + '\n' + cdstr
        else:
            sfstr = f"spray-frequency: {self.spray_frequency} Hz"
            return ptstr + '\n' + cdstr + '\n' + sfstr
        
    def add_meta(self, key, info):
        self._meta[key] = info


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ParameterFitResult:

    fitted_thickness: float
    params: np.ndarray
    inv_hessian: np.ndarray
    constrains: np.ndarray | None=None

    def __post_init__(self):
        assert len(self.params) == self.inv_hessian.shape[0]
        assert len(self.params) == self.inv_hessian.shape[1]
        self.dimension = len(self.params)

    @cached_property
    def hessian(self):
        return np.linalg.inv(self.inv_hessian)
    
    def is_hessian_regular(self) -> bool:
        if np.all(self.inv_hessian == np.diag(self.dimension)):
            return False
        return int(np.linalg.matrix_rank(self.inv_hessian)) == self.dimension

    def calculate_distance(self, point: np.ndarray) -> float:
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
    
    def did_it_terminate(self) -> bool:
        if np.all(self.inv_hessian == np.diag(self.dimension)):
            return False
        else:
            return True
        # np.linalg.diagonal(len(self.params)) == self.inv_hessian