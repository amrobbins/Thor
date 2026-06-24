from __future__ import annotations

import json
from pathlib import Path

import pytest

import thor


def _member_dir(root: Path, name: str) -> str:
    path = root / "members" / name
    path.mkdir(parents=True)
    return path.relative_to(root).as_posix()


def _minimal_manifest(root: Path, *, version: object = 1) -> dict[str, object]:
    return {
        "artifact_type": "thor_ensemble_model",
        "version": version,
        "execution": "parallel_single_gpu",
        "aggregation": {"type": "mean"},
        "input_names": ["matrix"],
        "output_names": ["prediction"],
        "reported_losses": [],
        "reported_metrics": [],
        "overall_loss_reduction": "sum",
        "losses": [],
        "metrics": [],
        "members": [
            {"name": "fold_0", "path": _member_dir(root, "fold_0"), "weight": 1.0, "selection": {}},
        ],
    }


def _write_manifest(root: Path, manifest: dict[str, object]) -> Path:
    manifest_path = root / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


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
        "reported_losses": [],
        "reported_metrics": [],
        "overall_loss_reduction": "sum",
        "losses": [],
        "metrics": [],
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
    assert loaded.reported_losses == ()
    assert loaded.reported_metrics == ()
    assert loaded.overall_loss_reduction == "sum"
    assert loaded.losses == ()
    assert loaded.metrics == ()
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


def test_ensemble_manifest_current_version_is_first_artifact_version():
    import thor.ensembles._manifest as manifest

    assert manifest._FIRST_ARTIFACT_VERSION == 1
    assert manifest._CURRENT_ARTIFACT_VERSION == manifest._FIRST_ARTIFACT_VERSION
    assert manifest._SUPPORTED_ARTIFACT_VERSIONS == frozenset({1})


def test_ensemble_model_load_accepts_first_artifact_version(tmp_path):
    _write_manifest(tmp_path, _minimal_manifest(tmp_path, version=1))

    loaded = thor.EnsembleModel.load(tmp_path)

    assert loaded.get_member_names() == ("fold_0",)
    assert loaded.get_input_names() == ("matrix",)
    assert loaded.get_output_names() == ("prediction",)


@pytest.mark.parametrize("bad_version", ["1", 1.0, True, None])
def test_ensemble_model_load_rejects_malformed_manifest_version(tmp_path, bad_version):
    _write_manifest(tmp_path, _minimal_manifest(tmp_path, version=bad_version))

    with pytest.raises(ValueError, match="version must be an integer"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_pre_first_manifest_version(tmp_path):
    _write_manifest(tmp_path, _minimal_manifest(tmp_path, version=0))

    with pytest.raises(ValueError, match="first supported version is 1"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_future_manifest_version(tmp_path):
    _write_manifest(tmp_path, _minimal_manifest(tmp_path, version=2))

    with pytest.raises(ValueError, match="current supported version is 1"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="ensemble manifest not found"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_invalid_json_manifest(tmp_path):
    (tmp_path / "ensemble_manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize("raw_manifest", [[], None, "not an object", 7])
def test_ensemble_model_load_rejects_non_object_manifest(tmp_path, raw_manifest):
    (tmp_path / "ensemble_manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ensemble manifest must be a JSON object"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize(
    "field_name",
    [
        "artifact_type",
        "version",
        "execution",
        "aggregation",
        "input_names",
        "output_names",
        "reported_losses",
        "reported_metrics",
        "overall_loss_reduction",
        "losses",
        "metrics",
        "members",
    ],
)
def test_ensemble_model_load_rejects_missing_required_manifest_fields(tmp_path, field_name):
    manifest = _minimal_manifest(tmp_path)
    manifest.pop(field_name)
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=f"missing required field {field_name!r}"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize("artifact_type", ["thor_model", None, 1])
def test_ensemble_model_load_rejects_bad_artifact_type(tmp_path, artifact_type):
    manifest = _minimal_manifest(tmp_path)
    manifest["artifact_type"] = artifact_type
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unsupported ensemble artifact_type"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_legacy_top_level_metric_policy_fields(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest["metric"] = "mae"
    manifest["target_input_name"] = "labels"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unsupported field"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize("execution", [None, 1, ["parallel_single_gpu"]])
def test_ensemble_model_load_rejects_non_string_execution(tmp_path, execution):
    manifest = _minimal_manifest(tmp_path)
    manifest["execution"] = execution
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="execution must be a string"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_unsupported_execution(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest["execution"] = "parallel_multi_gpu"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unsupported ensemble execution"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize(
    "aggregation, message",
    [
        (None, "aggregation must be a JSON object"),
        ([], "aggregation must be a JSON object"),
        ({}, "aggregation.type must be a string"),
        ({"type": 1}, "aggregation.type must be a string"),
        ({"type": "median"}, "unsupported ensemble aggregation"),
        ({"type": "mean", "legacy_metric": "mae"}, "unsupported field"),
    ],
)
def test_ensemble_model_load_rejects_malformed_aggregation(tmp_path, aggregation, message):
    manifest = _minimal_manifest(tmp_path)
    manifest["aggregation"] = aggregation
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize(
    "field_name, value, message",
    [
        ("input_names", "matrix", "input_names must be a sequence of strings"),
        ("output_names", "prediction", "output_names must be a sequence of strings"),
        ("input_names", ["matrix", "matrix"], "input_names entries must be unique"),
        ("output_names", [""], "output_names entries must be non-empty strings"),
        ("input_names", [1], "input_names entries must be non-empty strings"),
    ],
)
def test_ensemble_model_load_rejects_malformed_schema_name_lists(tmp_path, field_name, value, message):
    manifest = _minimal_manifest(tmp_path)
    manifest[field_name] = value
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize(
    "field_name, value, message",
    [
        ("reported_losses", "loss", "reported_losses must be a sequence of strings"),
        ("reported_losses", ["loss", "loss"], "reported_losses entries must be unique"),
        ("reported_losses", [""], "reported_losses entries must be non-empty strings"),
        ("reported_metrics", "mae", "reported_metrics must be a sequence of strings"),
        ("reported_metrics", ["mae", "mae"], "reported_metrics entries must be unique"),
        ("reported_metrics", [""], "reported_metrics entries must be non-empty strings"),
        ("overall_loss_reduction", "mean", "overall_loss_reduction must be 'sum'"),
        ("overall_loss_reduction", None, "overall_loss_reduction must be 'sum'"),
        ("losses", {}, "losses must be a JSON array"),
        ("losses", [{"name": "loss", "train_value": "bad", "test_value": None}], "train_value must be a finite number or null"),
        ("losses", [{"name": "loss", "train_value": None, "test_value": float("inf")}], "test_value must be a finite number or null"),
        ("losses", [{"name": "", "train_value": None, "test_value": None}], "name must be a non-empty string"),
        ("losses", [{"name": "loss", "train_value": None, "test_value": None, "target_input_name": "labels"}], "unsupported field"),
        ("metrics", {}, "metrics must be a JSON array"),
        ("metrics", [{"name": "mae", "train_value": "bad", "test_value": None}], "train_value must be a finite number or null"),
        ("metrics", [{"name": "mae", "train_value": None, "test_value": float("inf")}], "test_value must be a finite number or null"),
        ("metrics", [{"name": "", "train_value": None, "test_value": None}], "name must be a non-empty string"),
        ("metrics", [{"name": "mae", "train_value": None, "test_value": None, "target_input_name": "labels"}], "unsupported field"),
    ],
)
def test_ensemble_model_load_rejects_malformed_loss_reporting(tmp_path, field_name, value, message):
    manifest = _minimal_manifest(tmp_path)
    manifest[field_name] = value
    if field_name == "losses" and isinstance(value, list) and value and value[0].get("name") == "loss":
        manifest["reported_losses"] = ["loss"]
    if field_name == "metrics" and isinstance(value, list) and value and value[0].get("name") == "mae":
        manifest["reported_metrics"] = ["mae"]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_loss_reporting_mismatch(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest["reported_losses"] = ["mse_loss"]
    manifest["losses"] = [{"name": "mae_loss", "train_value": 1.0, "test_value": 2.0}]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unknown loss name"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_metric_reporting_mismatch(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest["reported_metrics"] = ["mse_metric"]
    manifest["metrics"] = [{"name": "mae_metric", "train_value": 1.0, "test_value": 2.0}]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="unknown metric name"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_legacy_manifest_without_loss_reporting(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest.pop("reported_losses")
    manifest.pop("overall_loss_reduction")
    manifest.pop("losses")
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="missing required field 'reported_losses'"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize("members", [None, {}, "fold_0"])
def test_ensemble_model_load_rejects_non_array_members(tmp_path, members):
    manifest = _minimal_manifest(tmp_path)
    manifest["members"] = members
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="members must be a JSON array"):
        thor.EnsembleModel.load(tmp_path)


def test_ensemble_model_load_rejects_empty_members(tmp_path):
    manifest = _minimal_manifest(tmp_path)
    manifest["members"] = []
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="at least one member"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize("member", [None, [], "fold_0"])
def test_ensemble_model_load_rejects_non_object_member_entries(tmp_path, member):
    manifest = _minimal_manifest(tmp_path)
    manifest["members"] = [member]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="member must be a JSON object"):
        thor.EnsembleModel.load(tmp_path)


@pytest.mark.parametrize(
    "member_patch, message",
    [
        ({"name": None}, "member.name must be a string"),
        ({"name": ""}, "member name must be a non-empty string"),
        ({"path": None}, "member 'fold_0' path must be a string"),
        ({"path": ""}, "path must not be empty"),
        ({"path": "/tmp/fold_0"}, "must be relative"),
        ({"path": "../fold_0"}, "must stay inside"),
        ({"weight": "1.0"}, "weight must be a finite positive number"),
        ({"weight": True}, "weight must be a finite positive number"),
        ({"weight": 0.0}, "weight must be a finite positive number"),
        ({"weight": -1.0}, "weight must be a finite positive number"),
        ({"weight": float("nan")}, "weight must be a finite positive number"),
        ({"selection": []}, "selection must be a mapping"),
        ({"target_input_name": "labels"}, "unsupported field"),
    ],
)
def test_ensemble_model_load_rejects_malformed_member_specs(tmp_path, member_patch, message):
    manifest = _minimal_manifest(tmp_path)
    member = dict(manifest["members"][0])
    member.update(member_patch)
    manifest["members"] = [member]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        thor.EnsembleModel.load(tmp_path)


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
        "reported_losses": [],
        "reported_metrics": [],
        "overall_loss_reduction": "sum",
        "losses": [],
        "metrics": [],
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



def test_ensemble_model_has_no_python_inference_runtime(tmp_path):
    fold_0 = _member_dir(tmp_path, "fold_0")
    model = thor.EnsembleModel([{"name": "fold_0", "path": fold_0}])

    assert not hasattr(model, "infer")
    assert not hasattr(model, "is_runtime_loaded")
    assert not hasattr(model, "unload_runtime")
    assert not hasattr(model, "_placed_members")
    assert not hasattr(model, "_placed_accumulator")
