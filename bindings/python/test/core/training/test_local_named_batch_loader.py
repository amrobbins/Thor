from __future__ import annotations

import json

import numpy as np
import pytest

import thor


def _layout() -> thor.training.LocalNamedExampleLayout:
    return thor.training.LocalNamedExampleLayout(
        tensors={
            "seasonality_inputs": [2],
            "monotone_inputs": [3],
            "daily_target": [2],
            "example_weight": [1],
        },
        data_type=thor.DataType.fp32,
    )


def _example(i: int) -> dict[str, np.ndarray]:
    base = float(i * 10)
    return {
        "seasonality_inputs": np.asarray([base + 1.0, base + 2.0], dtype=np.float32),
        "monotone_inputs": np.asarray([base + 3.0, base + 4.0, base + 5.0], dtype=np.float32),
        "daily_target": np.asarray([base + 6.0, base + 7.0], dtype=np.float32),
        "example_weight": np.asarray([base + 8.0], dtype=np.float32),
    }


def test_local_named_example_layout_constructs_packed_record_layout():
    layout = _layout()

    assert layout.get_data_type() == thor.DataType.fp32
    assert layout.get_record_size_bytes() == 32
    assert layout.get_tensor_names() == [
        "seasonality_inputs",
        "monotone_inputs",
        "daily_target",
        "example_weight",
    ]
    assert layout.get_tensor_shapes() == {
        "seasonality_inputs": [2],
        "monotone_inputs": [3],
        "daily_target": [2],
        "example_weight": [1],
    }
    assert layout.get_tensor_specs() == {
        "seasonality_inputs": {
            "shape": [2],
            "data_type": thor.DataType.fp32,
            "offset_bytes": 0,
            "num_bytes": 8,
        },
        "monotone_inputs": {
            "shape": [3],
            "data_type": thor.DataType.fp32,
            "offset_bytes": 8,
            "num_bytes": 12,
        },
        "daily_target": {
            "shape": [2],
            "data_type": thor.DataType.fp32,
            "offset_bytes": 20,
            "num_bytes": 8,
        },
        "example_weight": {
            "shape": [1],
            "data_type": thor.DataType.fp32,
            "offset_bytes": 28,
            "num_bytes": 4,
        },
    }


def test_local_named_writer_and_loader_round_trip_named_batches(tmp_path):
    dataset_path = tmp_path / "named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=2,
    )
    writer.write_example(_example(0), split="train")
    writer.write_example(_example(1), split="train")
    writer.write_example(_example(2), split="train")
    writer.write_example(_example(100), split="validate")

    assert writer.get_num_examples() == 4
    assert writer.get_num_train_examples() == 3
    assert writer.get_num_validate_examples() == 1
    assert writer.get_num_test_examples() == 0
    writer.close()
    assert writer.is_closed()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["format"] == "thor.local_named_example_dataset.v1"
    assert manifest["record_size_bytes"] == layout.get_record_size_bytes()
    assert manifest["num_examples"] == 4
    assert manifest["example_type_counts"] == {"train": 3, "validate": 1, "test": 0}
    assert len(manifest["shards"]) == 2

    loader = thor.training.LocalNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.get_dataset_name() == "local_named_examples"
    assert loader.get_batch_size() == 2
    assert loader.get_num_train_examples() == 3
    assert loader.get_num_validate_examples() == 1
    assert loader.get_num_test_examples() == 0
    assert loader.get_num_train_batches() == 2
    assert loader.get_num_validate_batches() == 1
    assert loader.get_num_test_batches() == 0
    assert loader.get_tensor_shapes() == layout.get_tensor_shapes()

    first_train_batch = loader.copy_next_batch("train")
    second_train_batch = loader.copy_next_batch("train")
    validate_batch = loader.copy_next_batch("validate")

    np.testing.assert_array_equal(
        first_train_batch["seasonality_inputs"],
        np.asarray([[1.0, 2.0], [11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first_train_batch["monotone_inputs"],
        np.asarray([[3.0, 4.0, 5.0], [13.0, 14.0, 15.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first_train_batch["daily_target"],
        np.asarray([[6.0, 7.0], [16.0, 17.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first_train_batch["example_weight"],
        np.asarray([[8.0], [18.0]], dtype=np.float32),
    )

    # Batch assembly wraps within the split exactly like the C++ loader path.
    np.testing.assert_array_equal(
        second_train_batch["seasonality_inputs"],
        np.asarray([[21.0, 22.0], [1.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        validate_batch["daily_target"],
        np.asarray([[1006.0, 1007.0], [1006.0, 1007.0]], dtype=np.float32),
    )


def test_local_named_loader_rejects_requested_layout_mismatch(tmp_path):
    dataset_path = tmp_path / "named_dataset"
    layout = _layout()
    writer = thor.training.LocalNamedExampleDatasetWriter(dataset_path, layout, examples_per_shard=10)
    writer.write_example(_example(0), split="train")
    writer.close()

    wrong_layout = thor.training.LocalNamedExampleLayout(
        tensors={
            "seasonality_inputs": [2],
            "monotone_inputs": [4],
            "daily_target": [2],
            "example_weight": [1],
        },
        data_type=thor.DataType.fp32,
    )

    with pytest.raises(RuntimeError, match="record_size_bytes|shape"):
        thor.training.LocalNamedBatchLoader(dataset_path, wrong_layout, batch_size=2)


def test_local_named_writer_rejects_missing_tensor(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(tmp_path / "named_dataset", _layout(), examples_per_shard=10)
    bad_example = _example(0)
    del bad_example["daily_target"]

    with pytest.raises(RuntimeError, match="tensor count|missing tensor"):
        writer.write_example(bad_example, split="train")


def test_local_named_writer_rejects_shape_mismatch(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(tmp_path / "named_dataset", _layout(), examples_per_shard=10)
    bad_example = _example(0)
    bad_example["daily_target"] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="shape"):
        writer.write_example(bad_example, split="train")


def test_local_named_writer_rejects_non_float32_or_non_contiguous_arrays(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(tmp_path / "named_dataset", _layout(), examples_per_shard=10)
    bad_example = _example(0)
    bad_example["daily_target"] = np.asarray([1.0, 2.0], dtype=np.float64)

    with pytest.raises(TypeError, match="C-contiguous numpy.float32"):
        writer.write_example(bad_example, split="train")

    bad_example = _example(0)
    non_contiguous = np.zeros((2, 2), dtype=np.float32)[:, 0]
    assert not non_contiguous.flags.c_contiguous
    bad_example["daily_target"] = non_contiguous

    with pytest.raises(TypeError, match="C-contiguous numpy.float32"):
        writer.write_example(bad_example, split="train")
