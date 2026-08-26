import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import thor


def _build_minimal_network() -> thor.Network:
    n = thor.Network("pytest_net")
    # Minimal graph: NetworkInput -> NetworkOutput
    ni = thor.layers.NetworkInput(n, "in", [1], thor.DataType.fp16)
    thor.layers.NetworkOutput(n, "out", ni.get_feature_output(), thor.DataType.fp16)
    return n


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _build_identity_network(name: str, dtype=thor.DataType.fp32) -> thor.Network:
    n = thor.Network(name)
    ni = thor.layers.NetworkInput(n, "input", [2], dtype)
    thor.layers.NetworkOutput(n, "output", ni.get_feature_output(), dtype)
    return n


def _build_identity_with_mse_training_outputs_network(
    name: str,
    *,
    include_prediction_output: bool,
) -> thor.Network:
    n = thor.Network(name)
    examples = thor.layers.NetworkInput(n, "examples", [1], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(n, "labels", [1], thor.DataType.fp32)
    loss = thor.losses.MSE(
        n,
        examples.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), thor.DataType.fp32)
    if include_prediction_output:
        thor.layers.NetworkOutput(n, "prediction", examples.get_feature_output(), thor.DataType.fp32)
    return n


@pytest.mark.cuda
def test_loaded_network_place_inference_only_infer_round_trip_identity(tmp_path):
    network_name = "pytest_loaded_inference_identity"
    values = np.array([[1.0, -2.0], [0.5, 3.0], [-4.0, 2.5]], dtype=np.float32)

    net = _build_identity_network(network_name, thor.DataType.fp32)
    placed = net.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    original_outputs = placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})
    assert set(original_outputs) == {"output"}
    assert np.allclose(original_outputs["output"].numpy(), values, atol=0.0)

    save_dir = tmp_path / "identity_saved_model"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert isinstance(loaded_placed, thor.runtime.PlacedNetwork)
    assert set(loaded_placed.get_network_input_names()) == {"input"}

    loaded_outputs = loaded_placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})

    assert set(loaded_outputs) == {"output"}
    assert np.allclose(loaded_outputs["output"].numpy(), values, atol=0.0)


@pytest.mark.cuda
def test_loaded_training_artifact_inference_prunes_indirect_loss_output_and_label_input(tmp_path):
    network_name = "pytest_loaded_inference_prunes_indirect_loss"
    examples_values = np.array([[1.0], [-2.0], [3.5]], dtype=np.float32)
    labels_values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

    net = _build_identity_with_mse_training_outputs_network(network_name, include_prediction_output=True)
    placed = net.place(
        examples_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    assert set(placed.get_network_input_names()) == {"examples", "labels"}
    in_memory_outputs = placed.infer(
        {
            "examples": _cpu_tensor(examples_values, thor.DataType.fp32),
            "labels": _cpu_tensor(labels_values, thor.DataType.fp32),
        }
    )
    assert set(in_memory_outputs) == {"prediction", "loss"}

    save_dir = tmp_path / "training_artifact_with_prediction_and_loss"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        examples_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert set(loaded_placed.get_network_input_names()) == {"examples"}
    loaded_outputs = loaded_placed.infer({"examples": _cpu_tensor(examples_values, thor.DataType.fp32)})
    assert set(loaded_outputs) == {"prediction"}
    assert np.allclose(loaded_outputs["prediction"].numpy(), examples_values, atol=0.0)

    with pytest.raises(RuntimeError, match=r"batchInputs.size\(\) == inputs.size\(\)"):
        loaded_placed.infer(
            {
                "examples": _cpu_tensor(examples_values, thor.DataType.fp32),
                "labels": _cpu_tensor(labels_values, thor.DataType.fp32),
            }
        )


@pytest.mark.cuda
def test_loaded_training_artifact_loss_only_graph_is_not_pruned(tmp_path):
    network_name = "pytest_loaded_inference_keeps_loss_only_graph"
    examples_values = np.array([[1.0], [-2.0], [3.5]], dtype=np.float32)
    labels_values = np.array([[0.5], [-1.5], [2.5]], dtype=np.float32)

    net = _build_identity_with_mse_training_outputs_network(network_name, include_prediction_output=False)
    placed = net.place(
        examples_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    save_dir = tmp_path / "loss_only_training_artifact"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        examples_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert set(loaded_placed.get_network_input_names()) == {"examples", "labels"}
    outputs = loaded_placed.infer(
        {
            "examples": _cpu_tensor(examples_values, thor.DataType.fp32),
            "labels": _cpu_tensor(labels_values, thor.DataType.fp32),
        }
    )
    assert set(outputs) == {"loss"}
    loss_value = np.array(outputs["loss"].numpy(), copy=True)
    assert loss_value.size >= 1
    assert np.all(np.isfinite(loss_value))

    with pytest.raises(RuntimeError):
        loaded_placed.infer({"examples": _cpu_tensor(examples_values, thor.DataType.fp32)})


@pytest.mark.cuda
def test_loaded_training_artifact_inference_pruning_keeps_independent_prediction_inputs(tmp_path):
    network_name = "pytest_loaded_inference_pruning_keeps_independent_prediction_inputs"
    features_values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    context_values = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    labels_values = np.array([[0.5], [1.5], [2.5]], dtype=np.float32)

    net = thor.Network(network_name)
    features = thor.layers.NetworkInput(net, "features", [1], thor.DataType.fp32)
    context = thor.layers.NetworkInput(net, "context", [1], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(net, "labels", [1], thor.DataType.fp32)
    loss = thor.losses.MSE(net, features.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "features_prediction", features.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "context_prediction", context.get_feature_output(), thor.DataType.fp32)

    placed = net.place(
        features_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    save_dir = tmp_path / "multi_input_independent_prediction_training_artifact"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        features_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert set(loaded_placed.get_network_input_names()) == {"features", "context"}
    outputs = loaded_placed.infer(
        {
            "features": _cpu_tensor(features_values, thor.DataType.fp32),
            "context": _cpu_tensor(context_values, thor.DataType.fp32),
        }
    )
    assert set(outputs) == {"features_prediction", "context_prediction"}
    assert np.allclose(outputs["features_prediction"].numpy(), features_values, atol=0.0)
    assert np.allclose(outputs["context_prediction"].numpy(), context_values, atol=0.0)

    with pytest.raises(RuntimeError, match=r"batchInputs.size\(\) == inputs.size\(\)"):
        loaded_placed.infer(
            {
                "features": _cpu_tensor(features_values, thor.DataType.fp32),
                "context": _cpu_tensor(context_values, thor.DataType.fp32),
                "labels": _cpu_tensor(labels_values, thor.DataType.fp32),
            }
        )


@pytest.mark.cuda
def test_loaded_training_artifact_inference_pruning_keeps_all_prediction_inputs(tmp_path):
    network_name = "pytest_loaded_inference_pruning_keeps_prediction_inputs"
    features_values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    context_values = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    labels_values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

    net = thor.Network(network_name)
    features = thor.layers.NetworkInput(net, "features", [1], thor.DataType.fp32)
    context = thor.layers.NetworkInput(net, "context", [1], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(net, "labels", [2], thor.DataType.fp32)
    prediction = thor.layers.Concatenate(
        net,
        [features.get_feature_output(), context.get_feature_output()],
        0,
    )
    loss = thor.losses.MSE(net, prediction.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(net, "prediction", prediction.get_feature_output(), thor.DataType.fp32)

    placed = net.place(
        features_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    save_dir = tmp_path / "multi_input_training_artifact"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        features_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert set(loaded_placed.get_network_input_names()) == {"features", "context"}
    outputs = loaded_placed.infer(
        {
            "features": _cpu_tensor(features_values, thor.DataType.fp32),
            "context": _cpu_tensor(context_values, thor.DataType.fp32),
        }
    )
    assert set(outputs) == {"prediction"}
    expected = np.concatenate([features_values, context_values], axis=1)
    assert np.allclose(outputs["prediction"].numpy(), expected, atol=0.0)


@pytest.mark.cuda
def test_concatenate_fp32_inference_matches_numpy():
    network_name = "pytest_concatenate_fp32_inference"
    left_values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    right_values = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

    net = thor.Network(network_name)
    left = thor.layers.NetworkInput(net, "left", [1], thor.DataType.fp32)
    right = thor.layers.NetworkInput(net, "right", [1], thor.DataType.fp32)
    concatenated = thor.layers.Concatenate(
        net,
        [left.get_feature_output(), right.get_feature_output()],
        0,
    )
    thor.layers.NetworkOutput(net, "concatenated", concatenated.get_feature_output(), thor.DataType.fp32)

    placed = net.place(
        left_values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "left": _cpu_tensor(left_values, thor.DataType.fp32),
            "right": _cpu_tensor(right_values, thor.DataType.fp32),
        }
    )

    assert set(outputs) == {"concatenated"}
    expected = np.concatenate([left_values, right_values], axis=1)
    assert np.allclose(outputs["concatenated"].numpy(), expected, atol=0.0)


@pytest.mark.cuda
def test_network_place_returns_placed_network():
    net = _build_minimal_network()

    placed = net.place(32)

    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert placed.get_network_name() == "pytest_net"
    assert placed.get_num_stamps() >= 1
    assert placed.get_num_trainable_layers() == net.get_num_trainable_layers()


@pytest.mark.cuda
def test_placed_network_save():
    with TemporaryDirectory(prefix="thor_pytest_") as tmp_path:
        net = _build_minimal_network()

        placed = net.place(7)

        out_dir = Path(tmp_path)
        placed.save(str(out_dir), overwrite=False, save_optimizer_state=False)

        assert out_dir.exists()
        assert out_dir.is_dir()
        assert out_dir.is_dir()
        archivePath = out_dir / "pytest_net.thor.tar"
        assert archivePath.exists()
        assert archivePath.is_file()


@pytest.mark.cuda
def test_placed_network_save_overwrite():
    with TemporaryDirectory(prefix="thor_pytest_") as tmp_path:
        net = _build_minimal_network()

        placed = net.place(50)

        out_dir = Path(tmp_path) / "placed_network"
        placed.save(str(out_dir), overwrite=False, save_optimizer_state=False)
        placed.save(str(out_dir), overwrite=True, save_optimizer_state=False)

        assert out_dir.exists()
        assert out_dir.is_dir()
        archivePath = out_dir / "pytest_net.thor.tar"
        assert archivePath.exists()
        assert archivePath.is_file()


@pytest.mark.cuda
def test_placed_network_basic_api(tmp_path):
    net = _build_minimal_network()

    placed = net.place(1)

    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert placed.get_network_name() == "pytest_net"

    num_stamps = placed.get_num_stamps()
    assert isinstance(num_stamps, int)
    assert num_stamps >= 1

    num_trainable_layers = placed.get_num_trainable_layers()
    assert isinstance(num_trainable_layers, int)
    assert num_trainable_layers == net.get_num_trainable_layers()

    save_dir = tmp_path / "saved"
    placed.save(str(save_dir), False, False)
    assert save_dir.exists()

    n = thor.Network("pytest_net")
    n.load(str(save_dir))
    placed2 = n.place(8)

    assert isinstance(placed2, thor.runtime.PlacedNetwork)
    assert placed2.get_network_name() == "pytest_net"

    num_stamps = placed2.get_num_stamps()
    assert isinstance(num_stamps, int)
    assert num_stamps >= 1

    num_trainable_layers = placed2.get_num_trainable_layers()
    assert isinstance(num_trainable_layers, int)
    assert num_trainable_layers == net.get_num_trainable_layers()


@pytest.mark.cuda
def test_ragged_network_output_infer_returns_one_logical_physical_ragged_tensor(tmp_path):
    batch_size = 2
    network_name = "pytest_ragged_network_output_infer"
    net = thor.Network(network_name)
    tokens = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    thor.layers.RaggedNetworkOutput(net, "tokens_out", tokens)

    placed = net.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    values_np = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [99.0, 99.0], [99.0, 99.0]],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 4], dtype=np.uint32)
    values = _cpu_tensor(values_np, thor.DataType.fp32)
    offsets = _cpu_tensor(offsets_np, thor.DataType.uint32)
    ragged = thor.physical.PhysicalRaggedTensor(values, offsets)

    outputs = placed.infer({"tokens": ragged})

    assert set(outputs) == {"tokens_out"}
    result = outputs["tokens_out"]
    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    active_values = int(offsets_np[-1])
    result_values = result.values.numpy()
    assert result_values.shape == values_np.shape
    assert np.array_equal(result_values[:active_values], values_np[:active_values])

    save_dir = tmp_path / "ragged_output_artifact"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    loaded_outputs = loaded_placed.infer({"tokens": ragged})
    assert set(loaded_outputs) == {"tokens_out"}
    loaded_result = loaded_outputs["tokens_out"]
    assert isinstance(loaded_result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(loaded_result.offsets.numpy(), offsets_np)
    loaded_values = loaded_result.values.numpy()
    assert loaded_values.shape == values_np.shape
    assert np.array_equal(loaded_values[:active_values], values_np[:active_values])


def test_physical_ragged_tensor_constructor_accepts_optional_max_values_per_row():
    values_np = np.zeros((8, 2), dtype=np.float32)
    offsets_np = np.array([0, 3, 5], dtype=np.uint32)
    values = _cpu_tensor(values_np, thor.DataType.fp32)
    offsets = _cpu_tensor(offsets_np, thor.DataType.uint32)

    unbounded = thor.physical.PhysicalRaggedTensor(values, offsets)
    assert unbounded.batch_size == 2
    assert unbounded.max_total_values == 8
    assert unbounded.max_values_per_row is None

    bounded = thor.physical.PhysicalRaggedTensor(values, offsets, max_values_per_row=4)
    assert bounded.batch_size == 2
    assert bounded.max_total_values == 8
    assert bounded.max_values_per_row == 4
    assert bounded.values is not None
    assert bounded.offsets is not None

    with pytest.raises(ValueError, match="max_values_per_row"):
        thor.physical.PhysicalRaggedTensor(values, offsets, max_values_per_row=0)


@pytest.mark.cuda
def test_physical_ragged_tensor_with_capacity_directly_infers_through_ragged_conv1d():
    batch_size = 2
    max_total_values = 8
    max_values_per_row = 4
    channels = 2

    net = thor.Network("pytest_physical_ragged_conv1d_direct_infer")
    x = thor.layers.RaggedNetworkInput(
        net,
        "tokens",
        thor.DataType.fp32,
        [channels],
        max_total_values=max_total_values,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=max_values_per_row,
    )
    conv = thor.layers.Convolution1d(
        net,
        x,
        num_output_channels=3,
        filter_width=3,
        stride=1,
        dilation=2,
        padding="causal",
        activation=None,
    )
    thor.layers.RaggedNetworkOutput(net, "encoded", conv.get_feature_output())

    placed = net.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    values_np = np.array(
        [
            [1.0, -1.0],
            [2.0, 0.5],
            [3.0, 1.5],
            [-2.0, 4.0],
            [0.25, -0.75],
            [99.0, 99.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 3, 5], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
        max_values_per_row=max_values_per_row,
    )

    outputs = placed.infer({"tokens": physical})

    assert set(outputs) == {"encoded"}
    encoded = outputs["encoded"]
    assert isinstance(encoded, thor.physical.PhysicalRaggedTensor)
    assert encoded.batch_size == batch_size
    assert encoded.max_total_values == max_total_values
    assert encoded.max_values_per_row == max_values_per_row
    assert np.array_equal(encoded.offsets.numpy(), offsets_np)
    assert encoded.values.numpy().shape == (max_total_values, 3)
    assert np.isfinite(encoded.values.numpy()[: int(offsets_np[-1])]).all()
