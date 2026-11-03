import pytest

import thor


def test_reduction_tuning_depth_enum_is_nested_and_exported_by_runtime():
    depth = thor.Network.ReductionTuningDepth
    assert thor.runtime.ReductionTuningDepth.regular is depth.regular
    assert {depth.quick, depth.regular, depth.ultra} == {
        thor.runtime.ReductionTuningDepth.quick,
        thor.runtime.ReductionTuningDepth.regular,
        thor.runtime.ReductionTuningDepth.ultra,
    }
    assert "ReductionTuningDepth" in dir(thor.runtime)
    assert not hasattr(thor, "ReductionTuningDepth")


def test_unset_reduction_tuning_depth_means_regular():
    network = thor.Network("tuning_depth_default")
    assert not network.is_reduction_tuning_depth_set()
    assert network.get_reduction_tuning_depth() is thor.Network.ReductionTuningDepth.regular


@pytest.mark.parametrize("name", ["quick", "regular", "ultra"])
def test_reduction_tuning_depth_round_trips_and_clears(name):
    depth = getattr(thor.Network.ReductionTuningDepth, name)
    network = thor.Network("tuning_depth_" + name)
    network.set_reduction_tuning_depth(depth)
    assert network.is_reduction_tuning_depth_set()
    assert network.get_reduction_tuning_depth() is depth
    network.set_reduction_tuning_depth(None)
    assert not network.is_reduction_tuning_depth_set()
    assert network.get_reduction_tuning_depth() is thor.Network.ReductionTuningDepth.regular


def test_reduction_tuning_depth_rejects_other_values():
    network = thor.Network("tuning_depth_bad")
    with pytest.raises(TypeError):
        network.set_reduction_tuning_depth("ultra")
