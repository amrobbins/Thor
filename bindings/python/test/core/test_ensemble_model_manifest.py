from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import thor


def _member_dir(root: Path, name: str) -> str:
    path = root / "members" / name
    path.mkdir(parents=True)
    return path.relative_to(root).as_posix()


def test_ensemble_model_saves_and_loads_manifest(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    fold_1 = _member_dir(tmp_path, "fold_1")

    model = thor.EnsembleModel(
        [
            thor.ensembles.EnsembleMemberSpec(
                name="fold_0",
                path=fold_0,
                selection={
                    "best_epoch": 3,
                    "best_score": 0.125,
                    "completed_epoch": 7,
                    "completion_reason": "early_completed",
                },
            ),
            {"name": "fold_1", "path": fold_1, "weight": 1.0},
        ],
        aggregation="mean",
        input_names=("trend_inputs", "seasonality_inputs", "monotone_increasing_inputs"),
        output_names=("forecast", "forecast_quantile_high"),
    )

    manifest_path = model.save(tmp_path, overwrite=True)
    assert manifest_path == tmp_path / "ensemble_manifest.json"

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw == {
        "artifact_type": "thor_ensemble_model",
        "version": 1,
        "execution": "parallel_single_gpu",
        "aggregation": {"type": "mean"},
        "input_names": ["trend_inputs", "seasonality_inputs", "monotone_increasing_inputs"],
        "output_names": ["forecast", "forecast_quantile_high"],
        "members": [
            {
                "name": "fold_0",
                "path": "members/fold_0",
                "weight": 1.0,
                "selection": {
                    "best_epoch": 3,
                    "best_score": 0.125,
                    "completed_epoch": 7,
                    "completion_reason": "early_completed",
                },
            },
            {"name": "fold_1", "path": "members/fold_1", "weight": 1.0, "selection": {}},
        ],
    }

    loaded = thor.EnsembleModel.load(tmp_path)
    assert loaded.artifact_path == tmp_path
    assert loaded.get_num_members() == 2
    assert loaded.get_member_names() == ("fold_0", "fold_1")
    assert loaded.get_member_paths() == ("members/fold_0", "members/fold_1")
    assert loaded.get_member_weights() == pytest.approx((1.0, 1.0))
    assert loaded.get_aggregation() == "mean"
    assert loaded.get_execution() == "parallel_single_gpu"
    assert loaded.get_input_names() == ("trend_inputs", "seasonality_inputs", "monotone_increasing_inputs")
    assert loaded.get_output_names() == ("forecast", "forecast_quantile_high")
    assert loaded.members[0].selection == {
        "best_epoch": 3,
        "best_score": 0.125,
        "completed_epoch": 7,
        "completion_reason": "early_completed",
    }


def test_ensemble_model_load_accepts_manifest_path(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    thor.EnsembleModel([{"name": "fold_0", "path": fold_0}]).save(tmp_path, overwrite=True)

    loaded = thor.EnsembleModel.load(tmp_path / "ensemble_manifest.json")

    assert loaded.artifact_path == tmp_path
    assert loaded.get_member_names() == ("fold_0",)


def test_ensemble_model_rejects_empty_ensemble():
    with pytest.raises(ValueError, match="at least one member"):
        thor.EnsembleModel([])


def test_ensemble_model_rejects_duplicate_member_names(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    with pytest.raises(ValueError, match="unique"):
        thor.EnsembleModel([
            {"name": "fold_0", "path": fold_0},
            {"name": "fold_0", "path": fold_0},
        ])


def test_ensemble_model_rejects_missing_member_path(tmp_path):
    manifest = {
        "artifact_type": "thor_ensemble_model",
        "version": 1,
        "execution": "parallel_single_gpu",
        "aggregation": {"type": "mean"},
        "input_names": [],
        "output_names": [],
        "members": [{"name": "fold_0", "path": "members/fold_0", "weight": 1.0, "selection": {}}],
    }
    (tmp_path / "ensemble_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="member artifact"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_rejects_absolute_or_escaping_member_paths(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        thor.ensembles.EnsembleMemberSpec(name="bad", path="/tmp/member")
    with pytest.raises(ValueError, match="inside"):
        thor.ensembles.EnsembleMemberSpec(name="bad", path="../member")


def test_ensemble_model_rejects_bad_aggregation_config(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    with pytest.raises(ValueError, match="unsupported ensemble aggregation"):
        thor.EnsembleModel([{"name": "fold_0", "path": fold_0}], aggregation="median")
    with pytest.raises(ValueError, match="weighted_mean"):
        thor.EnsembleModel([{"name": "fold_0", "path": fold_0, "weight": 0.5}], aggregation="mean")


def test_ensemble_model_supports_weighted_mean_manifest(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    fold_1 = _member_dir(tmp_path, "fold_1")
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": fold_0, "weight": 0.25},
            {"name": "fold_1", "path": fold_1, "weight": 0.75},
        ],
        aggregation="weighted_mean",
    ).save(tmp_path, overwrite=True)

    loaded = thor.EnsembleModel.load(tmp_path)

    assert loaded.get_aggregation() == "weighted_mean"
    assert loaded.get_member_weights() == pytest.approx((0.25, 0.75))


def test_ensemble_model_validates_schema_names(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    model = thor.EnsembleModel(
        [{"name": "fold_0", "path": fold_0}],
        input_names=["seasonality_inputs", "monotone_increasing_inputs"],
        output_names=["forecast"],
    )

    assert model.input_names == ("seasonality_inputs", "monotone_increasing_inputs")
    assert model.output_names == ("forecast",)
    with pytest.raises(ValueError, match="input_names entries must be unique"):
        thor.EnsembleModel([{"name": "fold_0", "path": fold_0}], input_names=["x", "x"])
    with pytest.raises(ValueError, match="output_names entries must be non-empty"):
        thor.EnsembleModel([{"name": "fold_0", "path": fold_0}], output_names=[""])


def test_ensemble_model_infer_requires_artifact_path(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    model = thor.EnsembleModel([{"name": "fold_0", "path": fold_0}])

    with pytest.raises(RuntimeError, match="artifact path"):
        model.infer({"x": object()})


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType = thor.DataType.fp32) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _build_transpose_member(name: str, *, transpose: bool) -> thor.Network:
    n = thor.Network(name)
    matrix = thor.layers.NetworkInput(n, "matrix", [2, 2], thor.DataType.fp32)
    output = matrix.get_feature_output()
    if transpose:
        output = thor.layers.Transpose(n, output).get_feature_output()
    thor.layers.NetworkOutput(n, "prediction", output, thor.DataType.fp32)
    return n


def _save_member_network(root: Path, name: str, *, transpose: bool, batch_size: int) -> str:
    network = _build_transpose_member(name, transpose=transpose)
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    member_path = root / "members" / name
    placed.save(str(member_path), overwrite=False, save_optimizer_state=False)
    return member_path.relative_to(root).as_posix()


def _build_multi_output_transpose_member(name: str, *, transpose: bool) -> thor.Network:
    n = thor.Network(name)
    matrix = thor.layers.NetworkInput(n, "matrix", [2, 2], thor.DataType.fp32)
    output = matrix.get_feature_output()
    if transpose:
        output = thor.layers.Transpose(n, output).get_feature_output()
    doubled = thor.layers.CustomLayer(
        network=n,
        inputs={"x": output},
        output_names=["scaled_prediction"],
        build=lambda context: {"scaled_prediction": context.input("x") * 2.0},
    )
    thor.layers.NetworkOutput(n, "prediction", output, thor.DataType.fp32)
    thor.layers.NetworkOutput(n, "scaled_prediction", doubled["scaled_prediction"], thor.DataType.fp32)
    return n


def _save_multi_output_member(root: Path, name: str, *, transpose: bool, batch_size: int) -> str:
    network = _build_multi_output_transpose_member(name, transpose=transpose)
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    member_path = root / "members" / name
    placed.save(str(member_path), overwrite=False, save_optimizer_state=False)
    return member_path.relative_to(root).as_posix()


def _build_member_with_output_name(
    name: str,
    *,
    output_name: str,
    input_shape: list[int] | tuple[int, ...] = (2, 2),
    transpose: bool = False,
    output_dtype: thor.DataType = thor.DataType.fp32,
) -> thor.Network:
    n = thor.Network(name)
    matrix = thor.layers.NetworkInput(n, "matrix", list(input_shape), thor.DataType.fp32)
    output = matrix.get_feature_output()
    if transpose:
        output = thor.layers.Transpose(n, output).get_feature_output()
    thor.layers.NetworkOutput(n, output_name, output, output_dtype)
    return n


def _save_custom_member(
    root: Path,
    name: str,
    *,
    batch_size: int,
    output_name: str = "prediction",
    input_shape: list[int] | tuple[int, ...] = (2, 2),
    transpose: bool = False,
    output_dtype: thor.DataType = thor.DataType.fp32,
) -> str:
    network = _build_member_with_output_name(
        name,
        output_name=output_name,
        input_shape=input_shape,
        transpose=transpose,
        output_dtype=output_dtype,
    )
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    member_path = root / "members" / name
    placed.save(str(member_path), overwrite=False, save_optimizer_state=False)
    return member_path.relative_to(root).as_posix()



def _build_demand_style_member(name: str) -> thor.Network:
    n = thor.Network(name)
    trend = thor.layers.NetworkInput(n, "trend_inputs", [1], thor.DataType.fp32)
    seasonality = thor.layers.NetworkInput(n, "seasonality_inputs", [2], thor.DataType.fp32)
    monotone = thor.layers.NetworkInput(n, "monotone_increasing_inputs", [1], thor.DataType.fp32)
    forecast = thor.layers.Concatenate(
        n,
        [
            trend.get_feature_output(),
            seasonality.get_feature_output(),
            monotone.get_feature_output(),
        ],
        0,
    )
    quantile = thor.layers.CustomLayer(
        network=n,
        inputs={"forecast": forecast.get_feature_output()},
        output_names=["forecast_quantile_high"],
        build=lambda context: {"forecast_quantile_high": context.input("forecast") + 1.0},
    )
    thor.layers.NetworkOutput(n, "forecast", forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(n, "forecast_quantile_high", quantile["forecast_quantile_high"], thor.DataType.fp32)
    return n


def _save_demand_style_member(root: Path, name: str, *, batch_size: int) -> str:
    network = _build_demand_style_member(name)
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    member_path = root / "members" / name
    placed.save(str(member_path), overwrite=False, save_optimizer_state=False)
    return member_path.relative_to(root).as_posix()


def test_network_input_alias_same_placement_inputs_constructor_flag():
    network = thor.Network("network-input-alias-flag")
    default_input = thor.layers.NetworkInput(network, "default", [2], thor.DataType.fp32)
    alias_input = thor.layers.NetworkInput(
        network,
        "alias",
        [2],
        thor.DataType.fp32,
        alias_same_placement_inputs=True,
    )

    assert not default_input.alias_same_placement_inputs()
    assert alias_input.alias_same_placement_inputs()


@pytest.mark.cuda
def test_network_outputs_on_gpu_placement_option_keeps_network_output_gpu_resident(tmp_path):
    batch_size = 1
    network = _build_transpose_member("gpu_output_member", transpose=False)
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
        network_outputs_on_gpu=True,
    )

    outputs = placed.infer({
        "matrix": _cpu_tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)),
    })

    prediction = outputs["prediction"]
    assert prediction.get_placement().get_device_type() == thor.physical.DeviceType.gpu
    assert prediction.get_placement().get_device_num() == 0
    with pytest.raises(ValueError, match="requires CPU placement"):
        prediction.numpy()

    cpu_prediction = prediction.clone(thor.physical.Placement(thor.physical.DeviceType.cpu, 0))
    stream = thor.physical.Stream(0)
    cpu_prediction.copy_from_async(prediction, stream)
    stream.synchronize()
    assert np.allclose(cpu_prediction.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), atol=0.0)


@pytest.mark.cuda
def test_ensemble_runtime_stages_named_inputs_once_to_gpu(tmp_path):
    batch_size = 2
    network = _build_transpose_member("input_staging_member", transpose=False)
    placed = network.place(
        batch_size,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
        network_outputs_on_gpu=True,
    )

    import thor._thor as _thor

    matrix = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    staged = _thor._stage_ensemble_inputs_once_for_debug(
        placed,
        {"matrix": _cpu_tensor(matrix)},
        0,
    )

    assert set(staged) == {"matrix"}
    staged_matrix = staged["matrix"]
    assert staged_matrix.get_placement().get_device_type() == thor.physical.DeviceType.gpu
    assert staged_matrix.get_placement().get_device_num() == 0

    cpu_copy = staged_matrix.clone(thor.physical.Placement(thor.physical.DeviceType.cpu, 0))
    stream = thor.physical.Stream(0)
    cpu_copy.copy_from_async(staged_matrix, stream)
    stream.synchronize()
    assert np.allclose(cpu_copy.numpy(), matrix, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_mean_aggregates_named_outputs(tmp_path, monkeypatch):
    batch_size = 3
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    member_1 = _save_member_network(tmp_path, "fold_1", transpose=True, batch_size=batch_size)
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        aggregation="mean",
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)

    ensemble = thor.EnsembleModel.load(tmp_path)

    import thor._thor as _thor

    def _old_python_aggregation_bridge_should_not_be_used(*args, **kwargs):
        raise AssertionError("EnsembleModel.infer should use the accumulator-network bridge")

    monkeypatch.setattr(
        _thor,
        "_aggregate_ensemble_outputs_gpu_to_cpu",
        _old_python_aggregation_bridge_should_not_be_used,
    )
    monkeypatch.setattr(
        _thor,
        "_infer_ensemble_members_and_aggregate_gpu_to_cpu",
        _old_python_aggregation_bridge_should_not_be_used,
    )

    matrix = np.array(
        [
            [[1.0, 10.0], [100.0, 1000.0]],
            [[2.0, 20.0], [200.0, 2000.0]],
            [[3.0, 30.0], [300.0, 3000.0]],
        ],
        dtype=np.float32,
    )

    outputs = ensemble.infer({"matrix": _cpu_tensor(matrix)})

    assert isinstance(outputs, dict)
    assert ensemble.is_runtime_loaded()
    assert getattr(ensemble, "_placed_accumulator") is not None
    assert set(outputs) == {"prediction"}
    assert outputs["prediction"].get_placement().get_device_type() == thor.physical.DeviceType.cpu
    expected = (matrix + np.swapaxes(matrix, 1, 2)) / 2.0
    assert np.allclose(outputs["prediction"].numpy(), expected, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_weighted_mean_aggregates_named_outputs(tmp_path):
    batch_size = 2
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    member_1 = _save_member_network(tmp_path, "fold_1", transpose=True, batch_size=batch_size)
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0, "weight": 1.0},
            {"name": "fold_1", "path": member_1, "weight": 3.0},
        ],
        aggregation="weighted_mean",
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)

    ensemble = thor.EnsembleModel.load(tmp_path)
    matrix = np.array(
        [
            [[1.0, 9.0], [90.0, 900.0]],
            [[4.0, 12.0], [120.0, 1200.0]],
        ],
        dtype=np.float32,
    )

    outputs = ensemble.infer({"matrix": _cpu_tensor(matrix)})

    assert outputs["prediction"].get_placement().get_device_type() == thor.physical.DeviceType.cpu
    expected = (matrix + 3.0 * np.swapaxes(matrix, 1, 2)) / 4.0
    assert np.allclose(outputs["prediction"].numpy(), expected, atol=0.0)




def test_ensemble_model_infer_runtime_does_not_use_thread_pool_executor():
    import thor.ensembles._manifest as manifest

    assert "ThreadPoolExecutor" not in inspect.getsource(manifest.EnsembleModel.infer)
    assert "ThreadPoolExecutor" not in inspect.getsource(manifest.EnsembleModel._infer_members_and_aggregate)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_reuses_resident_runtime_between_calls(tmp_path):
    batch_size = 2
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    member_1 = _save_member_network(tmp_path, "fold_1", transpose=True, batch_size=batch_size)
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        aggregation="mean",
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    first_matrix = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    second_matrix = first_matrix + 10.0

    first_outputs = ensemble.infer({"matrix": _cpu_tensor(first_matrix)})
    first_prediction = first_outputs["prediction"].numpy().copy()
    first_members = getattr(ensemble, "_placed_members")
    first_accumulator = getattr(ensemble, "_placed_accumulator")
    first_network = getattr(ensemble, "_accumulator_network")
    assert first_members is not None
    assert first_accumulator is not None
    assert first_network is not None

    second_outputs = ensemble.infer({"matrix": _cpu_tensor(second_matrix)})

    assert getattr(ensemble, "_placed_members") is first_members
    assert getattr(ensemble, "_placed_accumulator") is first_accumulator
    assert getattr(ensemble, "_accumulator_network") is first_network
    assert np.allclose(first_prediction, (first_matrix + np.swapaxes(first_matrix, 1, 2)) / 2.0, atol=0.0)
    assert np.allclose(second_outputs["prediction"].numpy(), (second_matrix + np.swapaxes(second_matrix, 1, 2)) / 2.0, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_aggregates_multiple_named_outputs_and_returns_cpu_tensors(tmp_path):
    batch_size = 2
    member_0 = _save_multi_output_member(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    member_1 = _save_multi_output_member(tmp_path, "fold_1", transpose=True, batch_size=batch_size)
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        aggregation="mean",
        input_names=["matrix"],
        output_names=["prediction", "scaled_prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)
    matrix = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )

    outputs = ensemble.infer({"matrix": _cpu_tensor(matrix)})

    expected = (matrix + np.swapaxes(matrix, 1, 2)) / 2.0
    assert set(outputs) == {"prediction", "scaled_prediction"}
    for output in outputs.values():
        assert output.get_placement().get_device_type() == thor.physical.DeviceType.cpu
    assert np.allclose(outputs["prediction"].numpy(), expected, atol=0.0)
    assert np.allclose(outputs["scaled_prediction"].numpy(), expected * 2.0, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_rejects_output_name_mismatch(tmp_path):
    batch_size = 1
    member_0 = _save_custom_member(tmp_path, "fold_0", batch_size=batch_size, output_name="prediction")
    member_1 = _save_custom_member(tmp_path, "fold_1", batch_size=batch_size, output_name="other_prediction")
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    with pytest.raises(RuntimeError, match="output names|prediction"):
        ensemble.infer({"matrix": _cpu_tensor(np.ones((1, 2, 2), dtype=np.float32))})


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_rejects_output_shape_mismatch(tmp_path):
    batch_size = 1
    member_0 = _save_custom_member(
        tmp_path,
        "fold_0",
        batch_size=batch_size,
        input_shape=(2, 3),
        transpose=False,
    )
    member_1 = _save_custom_member(
        tmp_path,
        "fold_1",
        batch_size=batch_size,
        input_shape=(2, 3),
        transpose=True,
    )
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    with pytest.raises(RuntimeError, match="descriptors do not match"):
        ensemble.infer({"matrix": _cpu_tensor(np.ones((1, 2, 3), dtype=np.float32))})


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_rejects_output_dtype_mismatch(tmp_path):
    batch_size = 1
    member_0 = _save_custom_member(
        tmp_path,
        "fold_0",
        batch_size=batch_size,
        output_dtype=thor.DataType.fp32,
    )
    member_1 = _save_custom_member(
        tmp_path,
        "fold_1",
        batch_size=batch_size,
        output_dtype=thor.DataType.fp64,
    )
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    with pytest.raises(RuntimeError, match="descriptors do not match"):
        ensemble.infer({"matrix": _cpu_tensor(np.ones((1, 2, 2), dtype=np.float32))})


@pytest.mark.cuda
def test_ensemble_model_infer_accepts_demand_style_inputs_and_returns_named_dict_outputs(tmp_path):
    batch_size = 2
    member_0 = _save_demand_style_member(tmp_path, "fold_0", batch_size=batch_size)
    thor.EnsembleModel(
        [{"name": "fold_0", "path": member_0}],
        aggregation="mean",
        input_names=["trend_inputs", "seasonality_inputs", "monotone_increasing_inputs"],
        output_names=["forecast", "forecast_quantile_high"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    trend = np.array([[1.0], [2.0]], dtype=np.float32)
    seasonality = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    monotone = np.array([[100.0], [200.0]], dtype=np.float32)

    outputs = ensemble.infer({
        "trend_inputs": _cpu_tensor(trend),
        "seasonality_inputs": _cpu_tensor(seasonality),
        "monotone_increasing_inputs": _cpu_tensor(monotone),
    })

    expected_forecast = np.concatenate([trend, seasonality, monotone], axis=1)
    assert isinstance(outputs, dict)
    assert set(outputs) == {"forecast", "forecast_quantile_high"}
    for output in outputs.values():
        assert output.get_placement().get_device_type() == thor.physical.DeviceType.cpu
    assert np.allclose(outputs["forecast"].numpy(), expected_forecast, atol=0.0)
    assert np.allclose(outputs["forecast_quantile_high"].numpy(), expected_forecast + 1.0, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_reloads_runtime_when_batch_size_changes(tmp_path):
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=1)
    member_1 = _save_member_network(tmp_path, "fold_1", transpose=True, batch_size=1)
    thor.EnsembleModel(
        [
            {"name": "fold_0", "path": member_0},
            {"name": "fold_1", "path": member_1},
        ],
        aggregation="mean",
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    first_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    second_matrix = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ],
        dtype=np.float32,
    )

    first_outputs = ensemble.infer({"matrix": _cpu_tensor(first_matrix)})
    first_members = getattr(ensemble, "_placed_members")
    first_accumulator = getattr(ensemble, "_placed_accumulator")

    second_outputs = ensemble.infer({"matrix": _cpu_tensor(second_matrix)})

    assert getattr(ensemble, "_placed_members") is not first_members
    assert getattr(ensemble, "_placed_accumulator") is not first_accumulator
    assert np.allclose(first_outputs["prediction"].numpy(), (first_matrix + np.swapaxes(first_matrix, 1, 2)) / 2.0, atol=0.0)
    assert np.allclose(second_outputs["prediction"].numpy(), (second_matrix + np.swapaxes(second_matrix, 1, 2)) / 2.0, atol=0.0)


@pytest.mark.cuda
def test_ensemble_model_parallel_infer_rejects_input_schema_mismatch(tmp_path):
    batch_size = 1
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    thor.EnsembleModel(
        [{"name": "fold_0", "path": member_0}],
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    with pytest.raises(ValueError, match=r"missing=\['matrix'\].*extra=\['missing_matrix'\]"):
        ensemble.infer({"missing_matrix": _cpu_tensor(np.ones((1, 2, 2), dtype=np.float32))})
    assert not ensemble.is_runtime_loaded()

    with pytest.raises(ValueError, match=r"extra=\['extra_matrix'\]"):
        ensemble.infer({
            "matrix": _cpu_tensor(np.ones((1, 2, 2), dtype=np.float32)),
            "extra_matrix": _cpu_tensor(np.ones((1, 2, 2), dtype=np.float32)),
        })
    assert not ensemble.is_runtime_loaded()


@pytest.mark.cuda
def test_ensemble_model_unload_runtime_releases_resident_members(tmp_path):
    batch_size = 1
    member_0 = _save_member_network(tmp_path, "fold_0", transpose=False, batch_size=batch_size)
    thor.EnsembleModel(
        [{"name": "fold_0", "path": member_0}],
        input_names=["matrix"],
        output_names=["prediction"],
    ).save(tmp_path, overwrite=True)
    ensemble = thor.EnsembleModel.load(tmp_path)

    ensemble.infer({
        "matrix": _cpu_tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)),
    })
    assert ensemble.is_runtime_loaded()

    ensemble.unload_runtime()

    assert not ensemble.is_runtime_loaded()
