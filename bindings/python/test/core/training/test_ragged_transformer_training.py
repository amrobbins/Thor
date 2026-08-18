import json
import numpy as np
import pytest

import thor


BATCH_SIZE = 3
FEATURES = 16
HISTORY_CAPACITY = 640
HISTORY_LENGTHS = np.asarray([371, 187, 54], dtype=np.uint32)
HISTORY_OFFSETS = np.asarray([0, 371, 558, 612], dtype=np.uint32)
HISTORY_BOUNDARY = 371
HISTORY_ORIGINS = np.asarray([[0], [184], [317]], dtype=np.int32)
FUTURE_LENGTH = 4
LEARNING_RATE = 0.05


def _constant(value: float):
    return thor.initializers.UniformRandom(value, value)


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


def _active_history_values() -> np.ndarray:
    active_rows = int(HISTORY_OFFSETS[-1])
    rows = np.arange(active_rows, dtype=np.int64)[:, None]
    features = np.arange(FEATURES, dtype=np.int64)[None, :]
    values = 0.35 * (((rows * 13 + features * 7) % 97).astype(np.float32) / 97.0)
    values += -0.17 + 0.002 * features.astype(np.float32)
    return np.ascontiguousarray(values.astype(np.float16))


def _future_values() -> np.ndarray:
    i = np.arange(BATCH_SIZE * FUTURE_LENGTH * FEATURES, dtype=np.int64)
    values = ((i * 11) % 71).astype(np.float32) / 90.0 - 0.28
    return np.ascontiguousarray(values.reshape(BATCH_SIZE, FUTURE_LENGTH, FEATURES).astype(np.float16))


def _future_labels() -> np.ndarray:
    i = np.arange(BATCH_SIZE * FUTURE_LENGTH * FEATURES, dtype=np.int64)
    values = 0.42 - ((i * 5) % 37).astype(np.float32) / 80.0
    return np.ascontiguousarray(values.reshape(BATCH_SIZE, FUTURE_LENGTH, FEATURES).astype(np.float16))


def _mirror_labels() -> np.ndarray:
    i = np.arange(BATCH_SIZE * FEATURES, dtype=np.int64)
    values = -0.33 + ((i * 3) % 29).astype(np.float32) / 45.0
    return np.ascontiguousarray(values.reshape(BATCH_SIZE, FEATURES).astype(np.float16))


def _build_network(name: str):
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
    future = thor.layers.NetworkInput(network, "future", [FUTURE_LENGTH, FEATURES], thor.DataType.fp16)
    history_origins = thor.layers.NetworkInput(network, "history_origins", [1], thor.DataType.int32)
    future_labels = thor.layers.NetworkInput(
        network, "future_labels", [FUTURE_LENGTH, FEATURES], thor.DataType.fp16
    )
    mirror_labels = thor.layers.NetworkInput(network, "mirror_labels", [FEATURES], thor.DataType.fp16)

    encoder_fc = thor.layers.FullyConnected(
        network,
        history,
        FEATURES * 2,
        True,
        activation=None,
        weights_initializer=_constant(0.035),
        biases_initializer=_constant(0.01),
    )
    encoded = thor.activations.Swiglu().add_to_network(network, encoder_fc.get_feature_output())
    encoder_norm = thor.layers.RMSNorm(
        network,
        encoded,
        epsilon=1e-5,
        weights_initializer=_constant(1.0),
    )
    encoded = thor.layers.DropOut(network, encoder_norm.get_feature_output(), 0.0).get_feature_output()
    encoded = thor.activations.Swish().add_to_network(network, encoded)

    attention_kwargs = dict(
        num_heads=1,
        head_dim=FEATURES,
        use_rope=True,
        rope_rotary_dim=FEATURES,
        weights_initializer=_constant(0.025),
        bias_initializer=_constant(0.0),
        dropout_probability=0.0,
    )
    self_attention = thor.layers.Attention(
        network,
        encoded,
        rope_query_position_offsets=history_origins.get_feature_output(),
        rope_key_position_offsets=history_origins.get_feature_output(),
        **attention_kwargs,
    )
    attention_history_states = self_attention.get_feature_output()
    assert isinstance(attention_history_states, thor.RaggedTensor)

    # Exercise the production transformer pattern where a ragged Attention
    # update is consumed immediately by a partition-preserving residual Add.
    # Both outputs share the query's structural row partition; packed values
    # carry no runtime partition state of their own.
    history_states = thor.layers.Add(
        network, encoded, attention_history_states
    ).get_feature_output()
    assert isinstance(history_states, thor.RaggedTensor)

    dense_ragged = thor.layers.Attention(
        network,
        future.get_feature_output(),
        context_input=history_states,
        rope_query_position_offset=HISTORY_BOUNDARY,
        rope_key_position_offsets=history_origins.get_feature_output(),
        **attention_kwargs,
    )
    assert dense_ragged.get_query_ragged() is False
    assert dense_ragged.get_key_value_ragged() is True

    ragged_dense = thor.layers.Attention(
        network,
        history_states,
        context_input=future.get_feature_output(),
        rope_query_position_offsets=history_origins.get_feature_output(),
        rope_key_position_offset=HISTORY_BOUNDARY,
        **attention_kwargs,
    )
    assert ragged_dense.get_query_ragged() is True
    assert ragged_dense.get_key_value_ragged() is False
    mirror_mean = thor.layers.SegmentedReduction(
        network,
        ragged_dense.get_feature_output(),
        thor.layers.SegmentedReduction.Type.mean,
    ).get_feature_output()

    future_loss = thor.losses.MSE(
        network,
        dense_ragged.get_feature_output(),
        future_labels.get_feature_output(),
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    mirror_loss = thor.losses.MSE(
        network,
        mirror_mean,
        mirror_labels.get_feature_output(),
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )

    thor.layers.NetworkOutput(network, "future_output", dense_ragged.get_feature_output(), thor.DataType.fp16)
    thor.layers.NetworkOutput(network, "mirror_mean", mirror_mean, thor.DataType.fp16)
    thor.layers.RaggedNetworkOutput(network, "encoded_history", history_states)
    thor.layers.NetworkOutput(network, "future_loss", future_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "mirror_loss", mirror_loss.get_loss(), thor.DataType.fp32)

    layers = {
        "encoder_fc": encoder_fc,
        "encoder_norm": encoder_norm,
        "self_attention": self_attention,
        "dense_ragged": dense_ragged,
        "ragged_dense": ragged_dense,
    }
    return network, layers


def _assert_no_ephemeral_row_partition_state_is_serialized(serialized: str):
    for forbidden in (
        "ragged" + "ActiveRows",
        "ragged_" + "active_rows",
        "hostActiveValueCount",
        "host_active_value_count",
        "rowPartitionHostActiveValueCount",
    ):
        assert forbidden not in serialized


def _assert_rope_and_mixed_modes(network: thor.Network):
    architecture_json = network.get_architecture_json()
    _assert_no_ephemeral_row_partition_state_is_serialized(architecture_json)
    architecture = json.loads(architecture_json)
    attentions = [layer for layer in architecture["layers"] if layer["layer_type"] == "attention"]
    assert len(attentions) == 3
    assert {(layer["query_ragged"], layer["key_value_ragged"]) for layer in attentions} == {
        (True, True),
        (False, True),
        (True, False),
    }

    dense_ragged = next(layer for layer in attentions if not layer["query_ragged"])
    assert dense_ragged["rope_query_position_offset"] == HISTORY_BOUNDARY
    assert dense_ragged["use_key_rope_position_offsets"] is True

    ragged_dense = next(layer for layer in attentions if not layer["key_value_ragged"])
    assert ragged_dense["use_query_rope_position_offsets"] is True
    assert ragged_dense["rope_key_position_offset"] == HISTORY_BOUNDARY

    expected_origins = HISTORY_BOUNDARY - HISTORY_LENGTHS.astype(np.int64)
    np.testing.assert_array_equal(HISTORY_ORIGINS[:, 0], expected_origins.astype(np.int32))


def test_ragged_transformer_training_phase_round_trip_preserves_only_structural_partition_state():
    network, _ = _build_network("pytest_ragged_transformer_training_phase")
    _assert_rope_and_mixed_modes(network)

    phase = thor.training.TrainingPhase("ragged_transformer_phase", network=network, enabled=True)
    phase_architecture = phase.get_architecture_json()
    _assert_no_ephemeral_row_partition_state_is_serialized(phase_architecture)

    restored = thor.training.TrainingPhase.deserialize(phase_architecture)
    assert restored.name == "ragged_transformer_phase"
    assert restored.enabled
    restored_network = restored.get_network()
    assert restored_network is not None
    _assert_rope_and_mixed_modes(restored_network)


def _inference_batch(poison: float):
    values = np.full((HISTORY_CAPACITY, FEATURES), np.float16(poison), dtype=np.float16)
    active = _active_history_values()
    values[: active.shape[0]] = active
    return {
        "history": _physical_ragged(values, HISTORY_OFFSETS),
        "future": _cpu_tensor(_future_values(), thor.DataType.fp16),
        "history_origins": _cpu_tensor(HISTORY_ORIGINS, thor.DataType.int32),
        "future_labels": _cpu_tensor(_future_labels(), thor.DataType.fp16),
        "mirror_labels": _cpu_tensor(_mirror_labels(), thor.DataType.fp16),
    }


def _dataset_and_data():
    dataset = thor.data.NumpyDataset(
        {
            "future": _future_values(),
            "history_origins": np.ascontiguousarray(HISTORY_ORIGINS),
            "future_labels": _future_labels(),
            "mirror_labels": _mirror_labels(),
        },
        ragged_tensors={
            "history": thor.data.RaggedBatch(
                _active_history_values(),
                np.ascontiguousarray(HISTORY_OFFSETS),
            )
        },
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(dataset=dataset, train_indices=[0, 1, 2], validate_indices=[]),
        batching=thor.data.BatchPolicy(batch_size=BATCH_SIZE, randomize_train=False),
        device_storage="off",
    )
    return dataset, data


def _infer(network: thor.Network, poison: float):
    placed = network.place(
        BATCH_SIZE,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    # A model saved from Trainer may mark label-only inputs as training-only.
    # Inference placement then prunes those inputs (and the corresponding loss
    # outputs), so the placed graph is the authoritative deployable boundary.
    available_inputs = _inference_batch(poison)
    required_input_names = set(placed.get_network_input_names())
    missing_inputs = required_input_names.difference(available_inputs)
    assert not missing_inputs, f"missing inference inputs: {sorted(missing_inputs)}"
    batch = {name: available_inputs[name] for name in required_input_names}

    outputs = placed.infer(batch)
    result = {}
    for name in ("future_output", "mirror_mean", "future_loss", "mirror_loss", "encoded_history"):
        if name not in outputs:
            continue
        value = outputs[name]
        if isinstance(value, thor.physical.PhysicalRaggedTensor):
            values = np.array(value.values.numpy(), copy=True)
            offsets = np.array(value.offsets.numpy(), copy=True)
            assert values.shape == (HISTORY_CAPACITY, FEATURES)
            np.testing.assert_array_equal(offsets, HISTORY_OFFSETS)
            active = int(offsets[-1])
            assert np.all(np.isfinite(values[:active])), name
            result[name] = (values, offsets)
        else:
            values = np.array(value.numpy(), copy=True)
            assert np.all(np.isfinite(values)), name
            result[name] = values

    assert "future_output" in result
    assert "mirror_mean" in result
    assert "encoded_history" in result
    return result


def _assert_outputs_close(lhs, rhs, *, atol=3e-2, rtol=3e-2):
    assert lhs.keys() == rhs.keys()
    for name in lhs:
        if isinstance(lhs[name], tuple):
            lhs_values, lhs_offsets = lhs[name]
            rhs_values, rhs_offsets = rhs[name]
            np.testing.assert_array_equal(lhs_offsets, rhs_offsets)
            active = int(lhs_offsets[-1])
            np.testing.assert_allclose(
                lhs_values[:active], rhs_values[:active], atol=atol, rtol=rtol, err_msg=name
            )
        else:
            np.testing.assert_allclose(lhs[name], rhs[name], atol=atol, rtol=rtol, err_msg=name)


def _changed(lhs: np.ndarray, rhs: np.ndarray, threshold: float = 1e-5) -> bool:
    return bool(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))) > threshold)


def _train_only(tmp_path, target: str):
    name = f"pytest_ragged_transformer_patch12_{target}"
    network, layers = _build_network(name)
    _assert_rope_and_mixed_modes(network)

    network.freeze_training()
    if target == "dense_ragged":
        layers["dense_ragged"].unfreeze_training()
    elif target == "ragged_dense":
        layers["ragged_dense"].unfreeze_training()
    elif target == "encoder":
        layers["encoder_fc"].unfreeze_training()
        layers["encoder_norm"].unfreeze_training()
        layers["self_attention"].unfreeze_training()
    else:
        raise AssertionError(target)

    dataset, data = _dataset_and_data()
    trainer = thor.training.Trainer(
        network=network,
        data=data,
        input_bindings=thor.training.DatasetInputBindings.by_exact_name(network=network, dataset=dataset),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=LEARNING_RATE, decay=0.0, momentum=0.0),
        debug_synchronous=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=[],
        stats_color="never",
    )
    result = trainer.fit(1, max_training_batches_per_epoch=1)
    assert result.status == "completed"

    save_dir = tmp_path / target
    trainer.save_model(str(save_dir), overwrite=False, save_optimizer_state=True)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    _assert_rope_and_mixed_modes(loaded)
    positive = _infer(loaded, 4096.0)
    negative = _infer(loaded, -4096.0)
    nan_poison = _infer(loaded, np.nan)
    _assert_outputs_close(positive, negative)
    _assert_outputs_close(positive, nan_poison)
    return positive


@pytest.mark.cuda
def test_ragged_transformer_both_mixed_quadrants_train_with_aligned_rope_and_poison_safe_capacity(tmp_path):
    baseline_network, _ = _build_network("pytest_ragged_transformer_patch12_baseline")
    _assert_rope_and_mixed_modes(baseline_network)

    # The public physical-ragged inference path lets us directly poison every row
    # outside offsets[-1]. Both mixed quadrants must be insensitive to that tail.
    positive_poison = _infer(baseline_network, 4096.0)
    negative_poison = _infer(baseline_network, -4096.0)
    nan_poison = _infer(baseline_network, np.nan)
    _assert_outputs_close(positive_poison, negative_poison)
    _assert_outputs_close(positive_poison, nan_poison)
    baseline = positive_poison

    # Isolate each mixed cross-attention quadrant. With every other trainable layer
    # frozen, a changed branch output after one Trainer batch is direct evidence that
    # backward reached and updated that mixed Attention layer.
    trained_dense_ragged = _train_only(tmp_path, "dense_ragged")
    assert _changed(baseline["future_output"], trained_dense_ragged["future_output"])

    trained_ragged_dense = _train_only(tmp_path, "ragged_dense")
    assert _changed(baseline["mirror_mean"], trained_ragged_dense["mirror_mean"])

    # Freeze both mixed Attention layers and train only the shared ragged encoder.
    # A changed downstream prediction proves finite/nonzero gradients traverse the
    # mixed cross-attention stages back into the ragged encoder.
    trained_encoder = _train_only(tmp_path, "encoder")
    assert _changed(baseline["future_output"], trained_encoder["future_output"]) or _changed(
        baseline["mirror_mean"], trained_encoder["mirror_mean"]
    )

    # _train_only() also reloads each trained artifact and verifies positive,
    # negative, and NaN poison inference equivalence. Save/reload and post-training
    # capacity safety are exercised for both mixed quadrants and the shared encoder.
