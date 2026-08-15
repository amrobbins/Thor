import json

import numpy as np
import pytest

import thor


BATCH_SIZE = 2
FEATURES = 16
HISTORY_CAPACITY = 7
FUTURE_LENGTH = 2
HISTORY_BOUNDARY = 371


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(values: np.ndarray, offsets: np.ndarray) -> thor.physical.PhysicalRaggedTensor:
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp16),
        _cpu_tensor(offsets, thor.DataType.uint32),
    )


def _build_network(name: str) -> thor.Network:
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp16,
        [FEATURES],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    future = thor.layers.NetworkInput(
        network,
        "future",
        [FUTURE_LENGTH, FEATURES],
        thor.DataType.fp16,
    )
    history_origins = thor.layers.NetworkInput(
        network,
        "history_origins",
        [1],
        thor.DataType.int32,
    )

    # Side statistic used by the forecaster: channel 0 -> per-product mean.
    channel0 = thor.layers.Slice(network, history, axis=0, start=0, length=1).get_feature_output()
    history_mean = thor.layers.SegmentedReduction(
        network,
        channel0,
        thor.layers.SegmentedReduction.Type.mean,
    ).get_feature_output()
    thor.layers.NetworkOutput(network, "history_mean", history_mean, thor.DataType.fp16)

    # Representative ragged transformer value path.
    wide = thor.layers.FullyConnected(network, history, FEATURES * 2, True, activation=None).get_feature_output()
    encoded = thor.activations.Swiglu().add_to_network(network, wide)
    encoded = thor.layers.RMSNorm(network, encoded, epsilon=1e-5).get_feature_output()
    encoded = thor.layers.DropOut(network, encoded, 0.2).get_feature_output()
    encoded = thor.activations.Swish().add_to_network(network, encoded)

    self_attention = thor.layers.Attention(
        network,
        encoded,
        1,
        head_dim=FEATURES,
        use_rope=True,
        rope_rotary_dim=FEATURES,
        rope_query_position_offsets=history_origins.get_feature_output(),
        rope_key_position_offsets=history_origins.get_feature_output(),
    )
    history_states = self_attention.get_feature_output()
    assert isinstance(history_states, thor.RaggedTensor)

    # Forecaster decoder path: dense future Q over ragged historical K/V.
    future_attention = thor.layers.Attention(
        network,
        future.get_feature_output(),
        1,
        head_dim=FEATURES,
        context_input=history_states,
        use_rope=True,
        rope_rotary_dim=FEATURES,
        rope_query_position_offset=HISTORY_BOUNDARY,
        rope_key_position_offsets=history_origins.get_feature_output(),
    )
    assert future_attention.get_query_ragged() is False
    assert future_attention.get_key_value_ragged() is True
    thor.layers.NetworkOutput(
        network,
        "future_output",
        future_attention.get_feature_output(),
        thor.DataType.fp16,
    )

    # Mirror mixed mode is part of the low/high-level completeness contract now.
    history_to_future = thor.layers.Attention(
        network,
        history_states,
        1,
        head_dim=FEATURES,
        context_input=future.get_feature_output(),
        use_rope=True,
        rope_rotary_dim=FEATURES,
        rope_query_position_offsets=history_origins.get_feature_output(),
        rope_key_position_offset=HISTORY_BOUNDARY,
    )
    assert history_to_future.get_query_ragged() is True
    assert history_to_future.get_key_value_ragged() is False
    mirror_mean = thor.layers.SegmentedReduction(
        network,
        history_to_future.get_feature_output(),
        thor.layers.SegmentedReduction.Type.mean,
    ).get_feature_output()
    thor.layers.NetworkOutput(network, "mirror_mean", mirror_mean, thor.DataType.fp16)

    # Exercise the canonical logical ragged output registry in the same artifact.
    thor.layers.RaggedNetworkOutput(network, "encoded_history", history_states)
    return network


def _batch_inputs():
    values = np.full((HISTORY_CAPACITY, FEATURES), np.float16(123.0), dtype=np.float16)
    active = 5
    values[:active] = (
        np.arange(active * FEATURES, dtype=np.float32).reshape(active, FEATURES) / 200.0 - 0.2
    ).astype(np.float16)
    offsets = np.array([0, 2, 5], dtype=np.uint32)
    history_origins = np.array([[HISTORY_BOUNDARY - 2], [HISTORY_BOUNDARY - 3]], dtype=np.int32)
    future = (
        np.arange(BATCH_SIZE * FUTURE_LENGTH * FEATURES, dtype=np.float32)
        .reshape(BATCH_SIZE, FUTURE_LENGTH, FEATURES)
        / 250.0
        - 0.1
    ).astype(np.float16)
    return {
        "history": _physical_ragged(values, offsets),
        "future": _cpu_tensor(future, thor.DataType.fp16),
        "history_origins": _cpu_tensor(history_origins, thor.DataType.int32),
    }, offsets


def _assert_composed_architecture(network: thor.Network):
    architecture = json.loads(network.get_architecture_json())
    layers = architecture["layers"]
    layer_types = [layer["layer_type"] for layer in layers]

    for required in ("slice", "fully_connected", "swiglu", "rms_norm", "drop_out", "swish"):
        assert required in layer_types
    assert layer_types.count("segmented_reduction") == 2

    attentions = [layer for layer in layers if layer["layer_type"] == "attention"]
    assert len(attentions) == 3
    modes = {(layer["query_ragged"], layer["key_value_ragged"]) for layer in attentions}
    assert modes == {(True, True), (False, True), (True, False)}
    assert all(layer["use_rope"] for layer in attentions)

    dense_q_ragged_kv = next(layer for layer in attentions if not layer["query_ragged"])
    assert dense_q_ragged_kv["rope_query_position_offset"] == HISTORY_BOUNDARY
    assert dense_q_ragged_kv["use_key_rope_position_offsets"] is True

    ragged_q_dense_kv = next(layer for layer in attentions if not layer["key_value_ragged"])
    assert ragged_q_dense_kv["rope_key_position_offset"] == HISTORY_BOUNDARY
    assert ragged_q_dense_kv["use_query_rope_position_offsets"] is True

    return architecture


def test_combined_ragged_transformer_python_surface_and_architecture_are_complete():
    network = _build_network("pytest_ragged_transformer_completeness_surface")
    _assert_composed_architecture(network)


@pytest.mark.cuda
def test_combined_ragged_transformer_placed_save_load_preserves_numerical_behavior(tmp_path):
    name = "pytest_ragged_transformer_completeness_save_load"
    network = _build_network(name)
    _assert_composed_architecture(network)

    placed = network.place(
        BATCH_SIZE,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    batch, expected_offsets = _batch_inputs()
    source_outputs = placed.infer(batch)
    assert set(source_outputs) == {"history_mean", "future_output", "mirror_mean", "encoded_history"}
    assert isinstance(source_outputs["encoded_history"], thor.physical.PhysicalRaggedTensor)
    np.testing.assert_array_equal(source_outputs["encoded_history"].offsets.numpy(), expected_offsets)

    save_dir = tmp_path / "ragged_transformer"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)

    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    _assert_composed_architecture(loaded)
    loaded_placed = loaded.place(
        BATCH_SIZE,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    loaded_batch, _ = _batch_inputs()
    loaded_outputs = loaded_placed.infer(loaded_batch)

    assert set(loaded_outputs) == set(source_outputs)
    for output_name in ("history_mean", "future_output", "mirror_mean"):
        np.testing.assert_allclose(
            np.array(loaded_outputs[output_name].numpy(), copy=True),
            np.array(source_outputs[output_name].numpy(), copy=True),
            rtol=2e-2,
            atol=2e-2,
        )

    source_history = source_outputs["encoded_history"]
    loaded_history = loaded_outputs["encoded_history"]
    np.testing.assert_array_equal(loaded_history.offsets.numpy(), source_history.offsets.numpy())
    np.testing.assert_allclose(
        np.array(loaded_history.values.numpy(), copy=True),
        np.array(source_history.values.numpy(), copy=True),
        rtol=2e-2,
        atol=2e-2,
    )
