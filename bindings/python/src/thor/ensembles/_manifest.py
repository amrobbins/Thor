"""Manifest-backed ensemble model artifact definitions.

The manifest is the stable saved-artifact contract.  The runtime keeps all
member networks resident after the first inference placement, stamps member
NetworkOutput tensors to GPU, submits ensemble member inference from native code,
and feeds the GPU-resident member outputs into a normal Thor accumulator network
that materializes the aggregated result.  The current accumulator uses normal
NetworkInput materialization because CustomLayer execution plans bind static
input tensors at placement time; the member outputs still remain GPU-resident
through aggregation and only the final accumulated result is copied to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence


_MANIFEST_FILENAME = "ensemble_manifest.json"
_SUPPORTED_ARTIFACT_TYPE = "thor_ensemble_model"
_SUPPORTED_VERSION = 1
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






def _name_mismatch_message(*, kind: str, expected: Sequence[str], actual: Sequence[str]) -> str:
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    parts = [
        f"ensemble {kind} names do not match",
        f"expected={list(expected)!r}",
        f"actual={list(actual)!r}",
    ]
    if missing:
        parts.append(f"missing={missing!r}")
    if extra:
        parts.append(f"extra={extra!r}")
    return "; ".join(parts)


def _validate_name_set(*, kind: str, expected: Sequence[str], actual: Sequence[str]) -> None:
    if set(actual) != set(expected):
        raise ValueError(_name_mismatch_message(kind=kind, expected=expected, actual=actual))


def _batch_size_from_inputs(batch_inputs: Mapping[str, object]) -> int:
    batch_size: int | None = None
    for name, tensor in batch_inputs.items():
        if not isinstance(name, str) or name == "":
            raise ValueError("batch input names must be non-empty strings")
        try:
            dims = tuple(tensor.get_dimensions())
        except AttributeError as exc:
            raise TypeError(f"batch input {name!r} must be a thor.physical.PhysicalTensor") from exc
        if not dims:
            raise ValueError(f"batch input {name!r} must have a leading batch dimension")
        current = int(dims[0])
        if batch_size is None:
            batch_size = current
        elif current != batch_size:
            raise ValueError(
                f"all ensemble batch inputs must have the same batch dimension; "
                f"input {name!r} has {current}, expected {batch_size}"
            )
    if batch_size is None or batch_size <= 0:
        raise ValueError("ensemble inference batch size must be positive")
    return batch_size





def _accumulator_input_name(output_index: int, member_index: int) -> str:
    return f"thor_ensemble_output_{output_index}_member_{member_index}"


def _normal_output_specs(raw_specs: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for output_name, raw in raw_specs.items():
        if not isinstance(output_name, str) or output_name == "":
            raise RuntimeError("ensemble member output names must be non-empty strings")
        dims = raw.get("dimensions")
        dtype = raw.get("data_type")
        if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes, bytearray)):
            raise RuntimeError(f"ensemble output {output_name!r} did not report dimensions")
        dim_tuple = tuple(int(d) for d in dims)
        if not dim_tuple:
            raise RuntimeError(f"ensemble output {output_name!r} must include a batch dimension")
        specs[output_name] = {"dimensions": dim_tuple, "data_type": dtype}
    return specs

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
        aggregation_type = mapping.get("type")
        if not isinstance(aggregation_type, str):
            raise ValueError("aggregation.type must be a string")
        return cls(type=aggregation_type)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


@dataclass(frozen=True)
class EnsembleMemberSpec:
    """One resident member model in an ensemble artifact manifest."""

    name: str
    path: str
    weight: float = 1.0
    selection: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValueError("member name must be a non-empty string")
        object.__setattr__(self, "path", _relative_path_string(self.path, field_name=f"member {self.name!r} path"))
        weight = float(self.weight)
        if weight <= 0.0:
            raise ValueError(f"member {self.name!r} weight must be positive")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "selection", _copy_json_object(self.selection, field_name=f"member {self.name!r} selection"))

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "EnsembleMemberSpec":
        mapping = _json_mapping(value, field_name="member")
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

    Ensemble artifacts are loadable, keep all member models resident during
    inference, and expose a dict-in/dict-out ``infer`` API over named Thor
    ``PhysicalTensor`` inputs and outputs.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMemberSpec | Mapping[str, Any]],
        *,
        aggregation: str | Mapping[str, Any] | EnsembleAggregation = "mean",
        execution: str = "parallel_single_gpu",
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
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
        self._artifact_path = Path(artifact_path) if artifact_path is not None else None
        self._runtime_lock = RLock()
        self._placed_members: tuple[object, ...] | None = None
        self._accumulator_network: object | None = None
        self._placed_accumulator: object | None = None
        self._runtime_output_names: tuple[str, ...] = tuple()
        self._runtime_batch_size: int | None = None
        self._runtime_device: int | None = None

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
            "version": _SUPPORTED_VERSION,
            "execution": self._execution,
            "aggregation": self._aggregation.to_dict(),
            "input_names": list(self._input_names),
            "output_names": list(self._output_names),
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
        artifact_type = manifest.get("artifact_type")
        if artifact_type != _SUPPORTED_ARTIFACT_TYPE:
            raise ValueError(
                f"unsupported ensemble artifact_type {artifact_type!r}; expected {_SUPPORTED_ARTIFACT_TYPE!r}"
            )
        version = manifest.get("version")
        if version != _SUPPORTED_VERSION:
            raise ValueError(f"unsupported ensemble manifest version {version!r}; expected {_SUPPORTED_VERSION}")
        execution = manifest.get("execution")
        if not isinstance(execution, str):
            raise ValueError("ensemble manifest execution must be a string")
        aggregation = EnsembleAggregation.from_user(manifest.get("aggregation", {"type": "mean"}))
        input_names = _string_tuple(manifest.get("input_names", []), field_name="ensemble manifest input_names")
        output_names = _string_tuple(manifest.get("output_names", []), field_name="ensemble manifest output_names")
        member_values = _json_sequence(manifest.get("members"), field_name="ensemble manifest members")
        members = tuple(EnsembleMemberSpec.from_manifest(_json_mapping(member, field_name="member")) for member in member_values)

        model = cls(
            members,
            aggregation=aggregation,
            execution=execution,
            input_names=input_names,
            output_names=output_names,
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

    def is_runtime_loaded(self) -> bool:
        """Return True when member models are already loaded and placed."""

        return self._placed_members is not None

    def unload_runtime(self) -> None:
        """Drop resident placed member networks so GPU memory can be released."""

        with self._runtime_lock:
            self._placed_members = None
            self._accumulator_network = None
            self._placed_accumulator = None
            self._runtime_output_names = tuple()
            self._runtime_batch_size = None
            self._runtime_device = None

    def infer(
        self,
        batch_inputs: Mapping[str, object],
        *,
        device: int = 0,
        reload_runtime: bool = False,
    ) -> dict[str, object]:
        """Run one parallel single-GPU ensemble inference batch.

        ``batch_inputs`` must contain CPU ``thor.physical.PhysicalTensor`` values
        keyed by network input name.  All member models are loaded and placed on
        first use, kept resident for subsequent calls with the same batch size,
        and submitted together through one native ensemble runtime bridge instead
        of Python worker threads.  Member ``NetworkOutput`` tensors are stamped to
        GPU.  When the producer is already on that GPU, ``NetworkOutput`` aliases
        the producer tensor instead of copying.  Those tensors then feed a normal
        Thor accumulator network.  The accumulator inputs intentionally use the
        ordinary NetworkInput materialization path for now because CustomLayer
        stamped execution plans bind their input tensor addresses when the
        accumulator is placed; dynamically aliasing different member-output
        tensors would require a deeper graph-composition contract.  The
        accumulator ``NetworkOutput`` materializes the final CPU tensors for
        this Python API.
        """

        if self._artifact_path is None:
            raise RuntimeError("EnsembleModel.infer requires an ensemble loaded from or saved to an artifact path")
        if self._execution != "parallel_single_gpu":
            raise RuntimeError(f"unsupported ensemble execution for infer: {self._execution!r}")
        if not isinstance(batch_inputs, Mapping):
            raise TypeError("batch_inputs must be a mapping from input name to PhysicalTensor")
        if not batch_inputs:
            raise ValueError("batch_inputs must not be empty")

        normalized_inputs = dict(batch_inputs)
        if self._input_names:
            _validate_name_set(kind="input", expected=self._input_names, actual=tuple(normalized_inputs.keys()))
        batch_size = _batch_size_from_inputs(normalized_inputs)
        placed_members = self._ensure_runtime_loaded(batch_size, device=device, reload_runtime=reload_runtime)
        self._validate_input_names(normalized_inputs, placed_members)

        return self._infer_members_and_aggregate(placed_members, normalized_inputs, device=int(device))

    def _ensure_runtime_loaded(self, batch_size: int, *, device: int, reload_runtime: bool) -> tuple[object, ...]:
        with self._runtime_lock:
            if (
                not reload_runtime
                and self._placed_members is not None
                and self._placed_accumulator is not None
                and self._runtime_batch_size == batch_size
                and self._runtime_device == device
            ):
                return self._placed_members

            self._validate_member_paths_exist()
            import thor  # Lazy import avoids a package-initialization cycle.

            placed_members = []
            for member in self._members:
                network = thor.Network(member.name)
                network.load(str(self._artifact_path / member.path))
                placed = network.place(
                    batch_size,
                    inference_only=True,
                    forced_devices=[int(device)],
                    forced_num_stamps_per_gpu=1,
                    network_outputs_on_gpu=True,
                )
                placed_members.append(placed)

            self._placed_members = tuple(placed_members)
            self._accumulator_network, self._placed_accumulator, self._runtime_output_names = self._build_accumulator_runtime(
                self._placed_members, batch_size=batch_size, device=int(device)
            )
            self._runtime_batch_size = batch_size
            self._runtime_device = int(device)
            return self._placed_members

    def _build_accumulator_runtime(
        self,
        placed_members: Sequence[object],
        *,
        batch_size: int,
        device: int,
    ) -> tuple[object, object, tuple[str, ...]]:
        if not placed_members:
            raise RuntimeError("ensemble runtime has no placed members")

        import thor  # Lazy import avoids a package-initialization cycle.
        import thor._thor as _thor  # type: ignore[import-not-found]

        raw_specs = _thor._get_ensemble_member_output_specs(
            placed_members[0], list(self._output_names)
        )
        output_specs = _normal_output_specs(raw_specs)
        output_names = tuple(output_specs.keys())
        if self._output_names and output_names != self._output_names:
            raise RuntimeError(
                f"ensemble output spec order mismatch: expected={list(self._output_names)!r} "
                f"actual={list(output_names)!r}"
            )

        weights = tuple(float(weight) for weight in self.get_member_weights())
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise RuntimeError("ensemble member weights must have a positive finite sum")
        inverse_weight_sum = 1.0 / weight_sum

        accumulator = thor.Network(f"{self.get_execution()}_ensemble_accumulator")
        for output_index, output_name in enumerate(output_names):
            spec = output_specs[output_name]
            dimensions = [int(d) for d in spec["dimensions"]]
            data_type = spec["data_type"]
            inputs: dict[str, object] = {}
            logical_names: list[str] = []
            for member_index in range(len(self._members)):
                input_name = _accumulator_input_name(output_index, member_index)
                logical_names.append(input_name)
                layer = thor.layers.NetworkInput(
                    accumulator,
                    input_name,
                    dimensions,
                    data_type,
                    dimensions_include_batch=True,
                    # Do not enable same-placement NetworkInput aliasing here yet.
                    # CustomLayer stamps expression input tensor addresses at
                    # accumulator placement time, so forwarding a different
                    # runtime tensor through NetworkInput leaves CustomLayer
                    # unable to match the arriving tensor to a connected input
                    # application.  Keeping the accumulator input materialized
                    # preserves the accumulator-network runtime while still
                    # avoiding member NetworkOutput CPU offload and copying only
                    # the final ensemble result back to CPU.
                    alias_same_placement_inputs=False,
                )
                inputs[input_name] = layer.get_feature_output()

            def build(context, *, _output_name=output_name, _input_names=tuple(logical_names), _weights=weights):
                expression = None
                for input_name, weight in zip(_input_names, _weights):
                    term = context.input(input_name) * float(weight)
                    expression = term if expression is None else expression + term
                if expression is None:
                    raise RuntimeError("ensemble accumulator requires at least one member input")
                return {_output_name: expression * inverse_weight_sum}

            layer = thor.layers.CustomLayer(
                network=accumulator,
                inputs=inputs,
                output_names=[output_name],
                build=build,
            )
            thor.layers.NetworkOutput(accumulator, output_name, layer[output_name], data_type)

        placed_accumulator = accumulator.place(
            batch_size,
            inference_only=True,
            forced_devices=[int(device)],
            forced_num_stamps_per_gpu=1,
            network_outputs_on_gpu=False,
        )
        return accumulator, placed_accumulator, output_names

    def _validate_input_names(self, batch_inputs: Mapping[str, object], placed_members: Sequence[object]) -> None:
        actual_names = tuple(batch_inputs.keys())
        if self._input_names:
            _validate_name_set(kind="input", expected=self._input_names, actual=actual_names)

        expected_from_members: set[str] | None = None
        for member, placed_member in zip(self._members, placed_members):
            member_names = set(placed_member.get_network_input_names())
            if expected_from_members is None:
                expected_from_members = member_names
            elif member_names != expected_from_members:
                raise RuntimeError(
                    f"ensemble member input names differ: member={member.name!r} "
                    f"input_names={sorted(member_names)!r} expected={sorted(expected_from_members)!r}"
                )
        if expected_from_members is not None and set(actual_names) != expected_from_members:
            raise ValueError(
                _name_mismatch_message(kind="input", expected=sorted(expected_from_members), actual=actual_names)
            )

    def _infer_members_and_aggregate(
        self,
        placed_members: Sequence[object],
        batch_inputs: Mapping[str, object],
        *,
        device: int,
    ) -> dict[str, object]:
        if not placed_members:
            raise RuntimeError("ensemble runtime has no placed members")

        if self._placed_accumulator is None:
            raise RuntimeError("ensemble accumulator runtime is not loaded")

        output_iteration_order = self._runtime_output_names or self._output_names or tuple()
        import thor._thor as _thor  # type: ignore[import-not-found]

        return _thor._infer_ensemble_members_then_accumulator_network(
            list(placed_members),
            self._placed_accumulator,
            dict(batch_inputs),
            list(output_iteration_order),
            int(device),
        )
