from pydantic.dataclasses import dataclass
from pydantic import (
    PositiveFloat, 
    confloat, 
    NonNegativeFloat,
)


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