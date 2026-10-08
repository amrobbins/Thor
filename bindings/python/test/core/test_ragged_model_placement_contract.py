"""Executable contracts for separating ragged model semantics from placement state."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import thor


_MAX_VALUES_PER_ROW = 4


def _build_neutral_ragged_network(name: str) -> thor.Network:
    network = thor.Network(name)
    values = thor.layers.RaggedNetworkInput(
        network,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_values_per_row=_MAX_VALUES_PER_ROW,
        offsets_data_type=thor.DataType.uint32,
    )
    thor.layers.RaggedNetworkOutput(network, "tokens_out", values)
    return network


def _read_saved_model_json(save_dir: Path, network_name: str) -> dict:
    archive_path = save_dir / f"{network_name}.thor.tar"
    assert archive_path.is_file(), f"expected Thor archive at {archive_path}"

    with tarfile.open(archive_path, mode="r:*") as archive:
        member = archive.extractfile(f"{network_name}.thor.json")
        assert member is not None
        return json.loads(member.read().decode("utf-8"))


def _paths_with_keys(value, forbidden_keys: set[str], path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in forbidden_keys:
                matches.append(child_path)
            matches.extend(_paths_with_keys(child, forbidden_keys, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_paths_with_keys(child, forbidden_keys, f"{path}[{index}]"))
    return matches


def test_ragged_network_input_model_contract_does_not_require_placement_capacity_or_batch():
    network = thor.Network("ragged_model_contract_without_placement_state")

    ragged = thor.layers.RaggedNetworkInput(
        network,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_values_per_row=_MAX_VALUES_PER_ROW,
        offsets_data_type=thor.DataType.uint32,
    )

    assert ragged.max_values_per_row == _MAX_VALUES_PER_ROW
    assert ragged.placement_neutral is True


def test_saved_ragged_artifact_omits_placement_derived_batch_and_capacity(tmp_path):
    network_name = "ragged_artifact_contract"
    network = _build_neutral_ragged_network(network_name)
    save_dir = tmp_path / "artifact"

    network.save(str(save_dir), overwrite=False)
    model_json = _read_saved_model_json(save_dir, network_name)

    forbidden_paths = _paths_with_keys(model_json, {"batch_size", "max_total_values"})

    network_inputs = {
        layer.get("name"): layer
        for layer in model_json.get("layers", [])
        if layer.get("layer_type") == "network_input"
    }
    values_input = network_inputs.get("tokens.values")
    partition_token = network_inputs.get("__thor_row_partition.tokens")
    derived_dimension_paths: list[str] = []
    if values_input is not None and values_input.get("dimensions") != [_MAX_VALUES_PER_ROW, 3]:
        derived_dimension_paths.append("$.layers[tokens.values].dimensions[0]")
    if partition_token is not None and partition_token.get("dimensions") != [1]:
        derived_dimension_paths.append("$.layers[__thor_row_partition.tokens].dimensions[0]")

    violations = forbidden_paths + derived_dimension_paths
    assert not violations, (
        "ragged model artifact contains placement-derived state: " + ", ".join(violations)
    )


@pytest.mark.cuda
def test_loaded_ragged_model_can_be_placed_repeatedly_at_different_batch_sizes(tmp_path):
    network_name = "ragged_cross_batch_placement_contract"
    network = _build_neutral_ragged_network(network_name)
    save_dir = tmp_path / "artifact"
    network.save(str(save_dir), overwrite=False)

    # Independently reconstruct the same artifact for each placement. Placement supplies B;
    # physical packed capacity and [B+1] partition extent are derived from B and the
    # model's max_values_per_row rather than inherited from the saved instance.
    for runtime_batch_size in (1, 3, 7):
        loaded = thor.Network(network_name)
        loaded.load(str(save_dir))
        placed = loaded.place(
            runtime_batch_size,
            inference_only=True,
            forced_devices=[0],
            forced_num_stamps_per_gpu=1,
        )
        assert placed.get_num_stamps() == 1
        del placed
