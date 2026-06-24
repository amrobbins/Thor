"""Manifest-backed ensemble model artifact definitions.

The manifest is the saved-artifact contract for TrainingRuns ensemble artifacts.
Training-produced ensemble artifacts store graph-owned reporting metadata:
``reported_losses`` names graph losses evaluated into ``losses``,
``reported_metrics`` names graph metrics evaluated into ``metrics``, and
``overall_loss_reduction`` records how the overall ensemble loss was reduced
from the graph loss values.  Legacy metric policy fields such as output-name,
target-input-name, or CPU-computed accuracy are intentionally not part of the
manifest schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping, Sequence


_MANIFEST_FILENAME = "ensemble_manifest.json"
_SUPPORTED_ARTIFACT_TYPE = "thor_ensemble_model"
_FIRST_ARTIFACT_VERSION = 1
_CURRENT_ARTIFACT_VERSION = _FIRST_ARTIFACT_VERSION
_SUPPORTED_ARTIFACT_VERSIONS = frozenset(range(_FIRST_ARTIFACT_VERSION, _CURRENT_ARTIFACT_VERSION + 1))
_SUPPORTED_EXECUTIONS = frozenset({"parallel_single_gpu"})
_SUPPORTED_AGGREGATIONS = frozenset({"mean", "weighted_mean"})


def _path_from_user(value: str | Path, *, field_name: str) -> Path:
    raw = str(value)
    if raw == "":
        raise ValueError(f"{field_name} must not be empty")
    return Path(raw)


def _json_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _json_sequence(value: object, *, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _required_manifest_field(manifest: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in manifest:
        raise ValueError(f"ensemble manifest missing required field {field_name!r}")
    return manifest[field_name]


def _reject_unknown_fields(mapping: Mapping[str, Any], *, allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported field(s): {unknown!r}")


def _relative_path_string(path: str | Path, *, field_name: str) -> str:
    raw = str(path)
    if raw == "":
        raise ValueError(f"{field_name} must not be empty")
    parsed = Path(raw)
    if parsed.is_absolute():
        raise ValueError(f"{field_name} must be relative to the ensemble artifact directory: {raw!r}")
    normalized = parsed.as_posix()
    if normalized == "." or normalized.startswith("../") or normalized == "..":
        raise ValueError(f"{field_name} must stay inside the ensemble artifact directory: {raw!r}")
    return normalized


def _copy_json_object(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when supplied")
    # Round-trip through json so callers cannot smuggle non-manifest objects into
    # the dataclass and so the error points at the public API boundary.
    try:
        return json.loads(json.dumps(dict(value)))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _json_nullable_number(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field_name} must be a finite number or null")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be a finite number or null")
    return out


def _named_value_result_from_manifest(value: Mapping[str, Any], *, field_name: str, index: int) -> dict[str, Any]:
    item_field = f"{field_name}[{index}]"
    mapping = _json_mapping(value, field_name=item_field)
    allowed = {"name", "train_value", "test_value"}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{item_field} contains unsupported field(s): {unknown!r}")
    name = mapping.get("name")
    if not isinstance(name, str) or name == "":
        raise ValueError(f"{item_field}.name must be a non-empty string")
    return {
        "name": name,
        "train_value": _json_nullable_number(mapping.get("train_value"), field_name=f"{item_field}.train_value"),
        "test_value": _json_nullable_number(mapping.get("test_value"), field_name=f"{item_field}.test_value"),
    }


def _named_value_results_tuple(value: Sequence[Mapping[str, Any]] | None, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return tuple()
    values = _json_sequence(value, field_name=field_name)
    results = tuple(
        _named_value_result_from_manifest(_json_mapping(item, field_name=f"{field_name}[{i}]"), field_name=field_name, index=i)
        for i, item in enumerate(values)
    )
    names = [result["name"] for result in results]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field_name} names must be unique; duplicates={duplicates!r}")
    return results


def _loss_results_tuple(value: Sequence[Mapping[str, Any]] | None, *, field_name: str) -> tuple[dict[str, Any], ...]:
    return _named_value_results_tuple(value, field_name=field_name)


def _metric_results_tuple(value: Sequence[Mapping[str, Any]] | None, *, field_name: str) -> tuple[dict[str, Any], ...]:
    return _named_value_results_tuple(value, field_name=field_name)


def _validate_named_reporting(
    *,
    reported_names: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    reported_field_name: str,
    result_field_name: str,
) -> None:
    result_names = tuple(result["name"] for result in results)
    missing = sorted(set(reported_names) - set(result_names))
    if missing:
        singular = {"losses": "loss", "metrics": "metric"}.get(result_field_name, result_field_name.rstrip("s"))
        raise ValueError(f"{reported_field_name} references unknown {singular} name(s): {missing!r}")
    unreported = sorted(set(result_names) - set(reported_names))
    if unreported:
        raise ValueError(f"{result_field_name} contains entry not listed in {reported_field_name}: {unreported!r}")


def _validate_loss_reporting(*, reported_losses: Sequence[str], losses: Sequence[Mapping[str, Any]]) -> None:
    _validate_named_reporting(
        reported_names=reported_losses,
        results=losses,
        reported_field_name="reported_losses",
        result_field_name="losses",
    )


def _validate_metric_reporting(*, reported_metrics: Sequence[str], metrics: Sequence[Mapping[str, Any]]) -> None:
    _validate_named_reporting(
        reported_names=reported_metrics,
        results=metrics,
        reported_field_name="reported_metrics",
        result_field_name="metrics",
    )




def _string_tuple(value: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    out = tuple(value)
    if any(not isinstance(item, str) or item == "" for item in out):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    if len(set(out)) != len(out):
        raise ValueError(f"{field_name} entries must be unique")
    return out


@dataclass(frozen=True)
class EnsembleAggregation:
    """How member predictions are combined by an ensemble artifact."""

    type: str = "mean"

    def __post_init__(self) -> None:
        if self.type not in _SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"unsupported ensemble aggregation {self.type!r}; "
                f"supported values are {sorted(_SUPPORTED_AGGREGATIONS)!r}"
            )

    @classmethod
    def from_user(cls, value: str | Mapping[str, Any] | "EnsembleAggregation" = "mean") -> "EnsembleAggregation":
        if isinstance(value, EnsembleAggregation):
            return value
        if isinstance(value, str):
            return cls(type=value)
        mapping = _json_mapping(value, field_name="aggregation")
        _reject_unknown_fields(mapping, allowed={"type"}, field_name="aggregation")
        aggregation_type = mapping.get("type")
        if not isinstance(aggregation_type, str):
            raise ValueError("aggregation.type must be a string")
        return cls(type=aggregation_type)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


@dataclass(frozen=True)
class EnsembleMemberSpec:
    """One member model entry in an ensemble artifact manifest."""

    name: str
    path: str
    weight: float = 1.0
    selection: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValueError("member name must be a non-empty string")
        object.__setattr__(self, "path", _relative_path_string(self.path, field_name=f"member {self.name!r} path"))
        if isinstance(self.weight, bool) or not isinstance(self.weight, numbers.Real):
            raise ValueError(f"member {self.name!r} weight must be a finite positive number")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"member {self.name!r} weight must be a finite positive number")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "selection", _copy_json_object(self.selection, field_name=f"member {self.name!r} selection"))

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "EnsembleMemberSpec":
        mapping = _json_mapping(value, field_name="member")
        _reject_unknown_fields(mapping, allowed={"name", "path", "weight", "selection"}, field_name="member")
        name = mapping.get("name")
        path = mapping.get("path")
        if not isinstance(name, str):
            raise ValueError("member.name must be a string")
        if not isinstance(path, str):
            raise ValueError(f"member {name!r} path must be a string")
        return cls(
            name=name,
            path=path,
            weight=mapping.get("weight", 1.0),
            selection=mapping.get("selection", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "weight": self.weight,
            "selection": dict(self.selection or {}),
        }


class EnsembleModel:
    """Manifest-backed Thor ensemble model artifact.

    Ensemble artifacts describe member selection and loss-reporting metadata.
    They no longer own a Python-side inference runtime; ensemble evaluation is
    performed by the composed graph-loss evaluator in TrainingRuns.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMemberSpec | Mapping[str, Any]],
        *,
        aggregation: str | Mapping[str, Any] | EnsembleAggregation = "mean",
        execution: str = "parallel_single_gpu",
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        reported_losses: Sequence[str] | None = None,
        reported_metrics: Sequence[str] | None = None,
        overall_loss_reduction: str = "sum",
        losses: Sequence[Mapping[str, Any]] | None = None,
        metrics: Sequence[Mapping[str, Any]] | None = None,
        artifact_path: str | Path | None = None,
    ) -> None:
        if execution not in _SUPPORTED_EXECUTIONS:
            raise ValueError(
                f"unsupported ensemble execution {execution!r}; supported values are {sorted(_SUPPORTED_EXECUTIONS)!r}"
            )
        member_specs = tuple(
            member if isinstance(member, EnsembleMemberSpec) else EnsembleMemberSpec.from_manifest(member)
            for member in members
        )
        if not member_specs:
            raise ValueError("ensemble must contain at least one member")
        names = [member.name for member in member_specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"ensemble member names must be unique; duplicates={duplicates!r}")

        aggregation_config = EnsembleAggregation.from_user(aggregation)
        if aggregation_config.type == "mean" and any(member.weight != 1.0 for member in member_specs):
            raise ValueError("non-unit member weights require aggregation='weighted_mean'")

        self._members = member_specs
        self._aggregation = aggregation_config
        self._execution = execution
        self._input_names = _string_tuple(input_names, field_name="input_names")
        self._output_names = _string_tuple(output_names, field_name="output_names")
        self._reported_losses = _string_tuple(reported_losses, field_name="reported_losses")
        self._reported_metrics = _string_tuple(reported_metrics, field_name="reported_metrics")
        if overall_loss_reduction != "sum":
            raise ValueError("overall_loss_reduction must be 'sum'")
        self._overall_loss_reduction = overall_loss_reduction
        self._losses = _loss_results_tuple(losses, field_name="losses")
        self._metrics = _metric_results_tuple(metrics, field_name="metrics")
        _validate_loss_reporting(reported_losses=self._reported_losses, losses=self._losses)
        _validate_metric_reporting(reported_metrics=self._reported_metrics, metrics=self._metrics)
        self._artifact_path = Path(artifact_path) if artifact_path is not None else None

    @property
    def artifact_path(self) -> Path | None:
        return self._artifact_path

    @property
    def execution(self) -> str:
        return self._execution

    @property
    def aggregation(self) -> EnsembleAggregation:
        return self._aggregation

    @property
    def members(self) -> tuple[EnsembleMemberSpec, ...]:
        return self._members

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def reported_losses(self) -> tuple[str, ...]:
        return self._reported_losses

    @property
    def reported_metrics(self) -> tuple[str, ...]:
        return self._reported_metrics

    @property
    def overall_loss_reduction(self) -> str:
        return self._overall_loss_reduction

    @property
    def losses(self) -> tuple[Mapping[str, Any], ...]:
        return self._losses

    @property
    def metrics(self) -> tuple[Mapping[str, Any], ...]:
        return self._metrics

    def get_num_members(self) -> int:
        return len(self._members)

    def get_member_names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self._members)

    def get_member_paths(self) -> tuple[str, ...]:
        return tuple(member.path for member in self._members)

    def get_member_weights(self) -> tuple[float, ...]:
        return tuple(member.weight for member in self._members)

    def get_aggregation(self) -> str:
        return self._aggregation.type

    def get_execution(self) -> str:
        return self._execution

    def get_input_names(self) -> tuple[str, ...]:
        return self._input_names

    def get_output_names(self) -> tuple[str, ...]:
        return self._output_names

    def to_manifest(self) -> dict[str, Any]:
        return {
            "artifact_type": _SUPPORTED_ARTIFACT_TYPE,
            "version": _CURRENT_ARTIFACT_VERSION,
            "execution": self._execution,
            "aggregation": self._aggregation.to_dict(),
            "input_names": list(self._input_names),
            "output_names": list(self._output_names),
            "reported_losses": list(self._reported_losses),
            "reported_metrics": list(self._reported_metrics),
            "overall_loss_reduction": self._overall_loss_reduction,
            "losses": [dict(loss) for loss in self._losses],
            "metrics": [dict(metric) for metric in self._metrics],
            "members": [member.to_dict() for member in self._members],
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write ``ensemble_manifest.json`` under ``path`` and return the file path."""

        artifact_dir = _path_from_user(path, field_name="path")
        manifest_path = artifact_dir / _MANIFEST_FILENAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"ensemble manifest already exists: {manifest_path}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._artifact_path = artifact_dir
        return manifest_path

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        """Load and validate an ensemble artifact directory or manifest path."""

        user_path = _path_from_user(path, field_name="path")
        manifest_path = user_path / _MANIFEST_FILENAME if user_path.is_dir() else user_path
        artifact_dir = manifest_path.parent
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"ensemble manifest not found: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"ensemble manifest is not valid JSON: {manifest_path}") from exc

        manifest = _json_mapping(raw_manifest, field_name="ensemble manifest")
        _reject_unknown_fields(
            manifest,
            allowed={
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
                "ensemble_group",
                "target_num_members",
                "actual_num_members",
                "min_successful_models",
            },
            field_name="ensemble manifest",
        )
        artifact_type = _required_manifest_field(manifest, "artifact_type")
        if artifact_type != _SUPPORTED_ARTIFACT_TYPE:
            raise ValueError(
                f"unsupported ensemble artifact_type {artifact_type!r}; expected {_SUPPORTED_ARTIFACT_TYPE!r}"
            )
        version = _required_manifest_field(manifest, "version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("ensemble manifest version must be an integer")
        if version < _FIRST_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported ensemble manifest version {version}; "
                f"first supported version is {_FIRST_ARTIFACT_VERSION}"
            )
        if version > _CURRENT_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported ensemble manifest version {version}; "
                f"current supported version is {_CURRENT_ARTIFACT_VERSION}"
            )
        if version not in _SUPPORTED_ARTIFACT_VERSIONS:
            raise ValueError(
                f"unsupported ensemble manifest version {version}; "
                f"supported versions are {sorted(_SUPPORTED_ARTIFACT_VERSIONS)!r}"
            )
        execution = _required_manifest_field(manifest, "execution")
        if not isinstance(execution, str):
            raise ValueError("ensemble manifest execution must be a string")
        aggregation = EnsembleAggregation.from_user(_required_manifest_field(manifest, "aggregation"))
        input_names = _string_tuple(
            _required_manifest_field(manifest, "input_names"),
            field_name="ensemble manifest input_names",
        )
        output_names = _string_tuple(
            _required_manifest_field(manifest, "output_names"),
            field_name="ensemble manifest output_names",
        )
        reported_losses = _string_tuple(
            _required_manifest_field(manifest, "reported_losses"),
            field_name="ensemble manifest reported_losses",
        )
        reported_metrics = _string_tuple(
            _required_manifest_field(manifest, "reported_metrics"),
            field_name="ensemble manifest reported_metrics",
        )
        overall_loss_reduction = _required_manifest_field(manifest, "overall_loss_reduction")
        if overall_loss_reduction != "sum":
            raise ValueError("ensemble manifest overall_loss_reduction must be 'sum'")
        losses = _loss_results_tuple(
            _required_manifest_field(manifest, "losses"),
            field_name="ensemble manifest losses",
        )
        metrics = _metric_results_tuple(
            _required_manifest_field(manifest, "metrics"),
            field_name="ensemble manifest metrics",
        )
        _validate_loss_reporting(reported_losses=reported_losses, losses=losses)
        _validate_metric_reporting(reported_metrics=reported_metrics, metrics=metrics)
        member_values = _json_sequence(
            _required_manifest_field(manifest, "members"),
            field_name="ensemble manifest members",
        )
        members = tuple(EnsembleMemberSpec.from_manifest(_json_mapping(member, field_name="member")) for member in member_values)

        model = cls(
            members,
            aggregation=aggregation,
            execution=execution,
            input_names=input_names,
            output_names=output_names,
            reported_losses=reported_losses,
            reported_metrics=reported_metrics,
            overall_loss_reduction=overall_loss_reduction,
            losses=losses,
            metrics=metrics,
            artifact_path=artifact_dir,
        )
        model._validate_member_paths_exist()
        return model

    def _validate_member_paths_exist(self) -> None:
        if self._artifact_path is None:
            return
        missing = [member.path for member in self._members if not (self._artifact_path / member.path).exists()]
        if missing:
            raise FileNotFoundError(f"ensemble member artifact path(s) do not exist: {missing!r}")
