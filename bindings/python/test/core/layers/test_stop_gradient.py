import json

import numpy as np
import pytest
import thor


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def test_stop_gradient_constructs_explicit_identity_layer():
    n = thor.Network("test_net_stop_gradient_constructs")
    x_in = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)

    stop = thor.layers.StopGradient(n, x_in.get_feature_output())

    assert isinstance(stop, thor.layers.StopGradient)
    y = stop.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == [4]
    assert y.get_data_type() == thor.DataType.fp32
    assert y != x_in.get_feature_output()

    arch = _only_layer_architecture(n, "stop_gradient")
    assert arch["feature_input"]["id"] == x_in.get_feature_output().get_id()
    assert arch["feature_output"]["id"] == y.get_id()
    assert arch["feature_input"]["id"] != arch["feature_output"]["id"]


@pytest.mark.cuda
def test_stop_gradient_forward_is_identity_alias():
    n = thor.Network("test_net_stop_gradient_forward_identity")
    dtype = thor.DataType.fp32
    x_in = thor.layers.NetworkInput(n, "input", [4], dtype)
    stop = thor.layers.StopGradient(n, x_in.get_feature_output())
    thor.layers.NetworkOutput(n, "output", stop.get_feature_output(), dtype)

    values = np.array([[1.0, -2.0, 3.5, 4.25], [-5.0, 6.0, -7.25, 8.5]], dtype=np.float32)
    placed = n.place(values.shape[0], inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    outputs = placed.infer({"input": _cpu_tensor(values, dtype)})

    assert set(outputs.keys()) == {"output"}
    np.testing.assert_allclose(np.array(outputs["output"].numpy(), copy=True), values, rtol=0.0, atol=0.0)


def test_stop_gradient_save_load_round_trip_preserves_explicit_layer(tmp_path):
    name = "test_net_stop_gradient_round_trip"
    n = thor.Network(name)
    dtype = thor.DataType.fp32
    x_in = thor.layers.NetworkInput(n, "input", [4], dtype)
    stop = thor.layers.StopGradient(n, x_in.get_feature_output())
    thor.layers.NetworkOutput(n, "output", stop.get_feature_output(), dtype)

    save_dir = tmp_path / "stop_gradient_model"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    arch = json.loads(loaded.get_architecture_json())
    assert sum(1 for layer in arch["layers"] if layer["layer_type"] == "stop_gradient") == 1


def test_stop_gradient_rejects_wrong_types_and_arity():
    n = thor.Network("test_net_stop_gradient_bad_args")
    x_in = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.layers.StopGradient()

    with pytest.raises(TypeError):
        thor.layers.StopGradient(n)

    with pytest.raises(TypeError):
        thor.layers.StopGradient("not a network", x_in.get_feature_output())

    with pytest.raises(TypeError):
        thor.layers.StopGradient(n, "not a tensor")
