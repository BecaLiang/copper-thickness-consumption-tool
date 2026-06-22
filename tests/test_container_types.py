import pytest


from copper_usage.feature_containers import (
    MachineFeatureContainer, 
    BoardFeatureContainer,
    ParameterFitResult,
)


def test_machine_features():

    mf1 = MachineFeatureContainer(
        plating_time=65,
        current_density=13.2,
        target_thickness=18,
    )
    assert mf1.current_density_range == (12.5, 13.5)

    mf1.current_density = 13.4
    assert mf1.current_density_range == (13.0, 14.0)

    mf1.current_density = 13.6
    assert mf1.current_density_range == (13.0, 14.0)

    mf1.current_density = 13.8
    assert mf1.current_density_range == (13.5, 14.5)