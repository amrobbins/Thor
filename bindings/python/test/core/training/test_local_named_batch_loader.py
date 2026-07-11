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

    assert writer.get_storage_mode() == "split"
    assert writer.get_num_examples() == 4
    assert writer.get_num_train_examples() == 3
    assert writer.get_num_validate_examples() == 1
    assert writer.get_num_test_examples() == 0
    writer.close()
    assert writer.is_closed()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["format"] == "thor.local_named_example_dataset.v1"
    assert manifest["storage_mode"] == "split"
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


def test_indexed_local_named_loader_reads_shared_dataset_by_indices(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=np.asarray([4, 2, 0], dtype=np.int64),
        validate_indices=np.asarray([1, 3], dtype=np.int64),
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.get_dataset_name() == "indexed_named_examples"
    assert loader.get_batch_size() == 2
    assert loader.get_num_dataset_examples() == 5
    assert loader.get_num_train_examples() == 3
    assert loader.get_num_validate_examples() == 2
    assert loader.get_num_test_examples() == 2
    assert loader.get_num_train_batches() == 2
    assert loader.get_num_validate_batches() == 1
    assert loader.get_num_test_batches() == 1
    assert not loader.has_explicit_test_split()
    assert loader.get_tensor_shapes() == layout.get_tensor_shapes()

    train_batch = loader.copy_next_batch("train")
    validate_batch = loader.copy_next_batch("validate")
    test_batch = loader.copy_next_batch("test")

    np.testing.assert_array_equal(
        train_batch["seasonality_inputs"],
        np.asarray([[41.0, 42.0], [21.0, 22.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        train_batch["daily_target"],
        np.asarray([[46.0, 47.0], [26.0, 27.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        validate_batch["seasonality_inputs"],
        np.asarray([[11.0, 12.0], [31.0, 32.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(test_batch["seasonality_inputs"], validate_batch["seasonality_inputs"])


def test_local_named_dataset_owns_schema_identity_and_shared_reader(tmp_path):
    dataset_path = tmp_path / "immutable_indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    with (dataset_path / "manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    assert manifest["dataset_id"] == str(writer.get_dataset_id())

    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    reopened = thor.data.LocalNamedDataset.open(dataset_path)
    assert str(dataset.id) == manifest["dataset_id"]
    assert dataset.id == reopened.id
    assert dataset.num_examples == 5
    assert dataset.schema.names == [
        "seasonality_inputs",
        "monotone_inputs",
        "daily_target",
        "example_weight",
    ]
    assert dataset.field("seasonality_inputs").dtype == thor.DataType.fp32
    assert dataset.field("seasonality_inputs").shape == [2]
    assert dataset.field("seasonality_inputs").kind == thor.data.DatasetFieldKind.DENSE
    assert dataset.schema.contains("seasonality_inputs")
    assert dataset.schema.field("seasonality_inputs").name == "seasonality_inputs"
    assert dataset.schema["seasonality_inputs"].name == "seasonality_inputs"

    fold_a = thor.training.IndexedNamedBatchLoader(
        dataset=dataset,
        train_indices=[0, 2, 4],
        validate_indices=[1],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )
    fold_b = thor.training.IndexedNamedBatchLoader(
        dataset=dataset,
        train_indices=[1, 3],
        validate_indices=[0, 4],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert fold_a.get_dataset() is dataset
    assert fold_b.get_dataset() is dataset
    np.testing.assert_array_equal(
        fold_a.copy_next_batch("train")["seasonality_inputs"],
        np.asarray([[1.0, 2.0], [21.0, 22.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fold_b.copy_next_batch("train")["seasonality_inputs"],
        np.asarray([[11.0, 12.0], [31.0, 32.0]], dtype=np.float32),
    )


def test_indexed_local_named_loader_exposes_stats(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=[0, 1, 2, 3, 4],
        validate_indices=[],
        test_indices=[],
        batch_size=1,
        batch_queue_depth=2,
        randomize_train=False,
    )

    before = loader.get_stats("train")
    assert before["split"] == "train"
    assert before["target_batch_queue_depth"] == 2
    assert before["records_requested"] >= 0
    assert before["logical_record_bytes_requested"] >= 0
    assert before["read_calls_submitted"] >= 0
    assert before["read_bytes_submitted"] >= 0
    assert before["read_calls_completed"] >= 0
    assert before["read_bytes_completed"] >= 0
    assert before["records_copied"] >= 0
    assert before["record_copy_bytes"] >= 0
    assert before["record_copy_memcpy_calls"] >= 0
    assert before["record_copy_active_nanoseconds"] >= 0
    assert before["record_copy_pop_wait_nanoseconds"] >= 0
    assert before["completed_record_queue_push_wait_nanoseconds"] >= 0
    assert before["copied_record_queue_push_wait_nanoseconds"] >= 0
    # The indexed loader now performs direct slice reads into named batch tensor
    # memory. There is no staging record-buffer pool in this path.
    assert before["record_buffer_pool_capacity"] == 0
    assert before["current_record_buffer_pool_depth"] == 0
    assert before["batches_assembled"] >= 0
    assert before["batches_delivered"] == 0
    assert before["batch_buffers_returned"] == 0
    assert before["current_ready_batches"] >= 0
    assert before["current_pending_batches"] >= 0
    assert before["current_completed_record_queue_depth"] >= 0
    assert before["current_copied_record_queue_depth"] >= 0
    assert before["shard_read_queue_depth"] >= 1
    assert before["shard_request_queue_depth"] >= 1
    assert before["completed_record_queue_depth"] >= 1
    # Direct reads eliminated copy-worker threads; loader-owned read workers
    # load directly into tensor slots.
    assert before["record_copy_thread_count"] == 0
    assert before["record_size_bytes"] == layout.get_record_size_bytes()
    assert isinstance(before["resolved_io_backend"], str)
    assert before["read_amplification"] >= 0.0
    assert before["planning_lead_records"] >= 0.0
    assert before["average_copy_nanoseconds_per_record"] >= 0.0
    assert before["average_copy_memcpy_calls_per_record"] >= 0.0
    assert before["average_copy_bytes_per_record"] >= 0.0
    diagnostic_keys = [
        "get_batch_wait_nanoseconds",
        "get_batch_tensor_unload_wait_nanoseconds",
        "load_worker_read_submit_nanoseconds",
        "load_worker_read_drain_nanoseconds",
        "readv_completion_wait_nanoseconds",
        "reader_drain_nanoseconds",
        "reader_drain_context_visits",
        "reader_drain_submit_calls",
        "reader_drain_submit_nanoseconds",
        "reader_drain_wait_loop_nanoseconds",
        "reader_drain_completion_process_nanoseconds",
        "reader_drain_completions",
        "reader_drain_max_inflight_reads",
        "reader_load_example_calls",
        "reader_load_example_nanoseconds",
        "reader_resolve_shard_nanoseconds",
        "reader_shard_context_lookup_calls",
        "reader_shard_context_cache_hits",
        "reader_shard_context_cache_misses",
        "reader_shard_context_lookup_nanoseconds",
        "reader_shard_read_request_nanoseconds",
        "reader_iovec_slot_acquire_nanoseconds",
        "reader_iovec_fill_nanoseconds",
        "reader_readv_submit_call_nanoseconds",
        "start_batch_tensor_acquire_nanoseconds",
        "start_batch_planning_nanoseconds",
        "oldest_pending_batch_age_nanoseconds",
        "average_pending_batch_age_nanoseconds",
        "current_pending_loaded_batches",
        "current_pending_unloaded_batches",
    ]
    for key in diagnostic_keys:
        assert before[key] >= 0

    batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(
        batch["seasonality_inputs"],
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )

    after = loader.get_stats("train")
    assert after["split"] == "train"
    record_size = layout.get_record_size_bytes()
    assert after["records_requested"] >= 1
    assert after["logical_record_bytes_requested"] >= record_size
    assert after["logical_record_bytes_requested"] % record_size == 0
    assert after["read_calls_submitted"] >= 1
    assert after["read_bytes_submitted"] >= record_size
    assert after["read_bytes_submitted"] % record_size == 0
    assert after["read_calls_completed"] >= 1
    assert after["read_bytes_completed"] >= record_size
    assert after["read_bytes_completed"] % record_size == 0
    assert after["records_copied"] == 0
    assert after["record_copy_bytes"] == 0
    # Direct vectorized reads avoid the CPU memcpy fanout stage entirely.
    assert after["record_copy_memcpy_calls"] == 0
    assert after["average_copy_bytes_per_record"] == 0.0
    assert after["average_copy_memcpy_calls_per_record"] == 0.0
    assert after["read_amplification"] == 1.0
    assert "readv" in after["resolved_io_backend"]
    assert after["batches_assembled"] >= 1
    assert after["batches_delivered"] >= 1
    assert after["batch_buffers_returned"] >= 1

    empty_stats = loader.get_stats("validate")
    assert empty_stats["split"] == "validate"
    assert empty_stats["records_requested"] == 0
    assert empty_stats["target_batch_queue_depth"] == 2
    assert empty_stats["record_copy_thread_count"] == 0
    assert empty_stats["record_buffer_pool_capacity"] == 0
    assert empty_stats["record_size_bytes"] == layout.get_record_size_bytes()
    assert empty_stats["resolved_io_backend"] == "empty"


def test_indexed_local_named_loader_allows_empty_validate_and_test_indices(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    empty = np.empty((0,), dtype=np.int64)
    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        validate_indices=empty,
        test_indices=empty,
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.has_explicit_test_split()
    assert loader.get_num_train_examples() == 3
    assert loader.get_num_validate_examples() == 0
    assert loader.get_num_test_examples() == 0
    assert loader.get_num_train_batches() == 2
    assert loader.get_num_validate_batches() == 0
    assert loader.get_num_test_batches() == 0

    train_batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(
        train_batch["seasonality_inputs"],
        np.asarray([[1.0, 2.0], [11.0, 12.0]], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="empty split"):
        loader.copy_next_batch("validate")

    with pytest.raises(RuntimeError, match="empty split"):
        loader.copy_next_batch("test")


def test_indexed_local_named_loader_empty_validate_aliases_empty_implicit_test(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(3):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=[0, 1, 2],
        validate_indices=np.empty((0,), dtype=np.int64),
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert not loader.has_explicit_test_split()
    assert loader.get_num_validate_examples() == 0
    assert loader.get_num_test_examples() == 0
    assert loader.get_num_validate_batches() == 0
    assert loader.get_num_test_batches() == 0


def test_indexed_local_named_loader_still_rejects_empty_train_indices(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    writer.write_indexed_example(_example(0))
    writer.close()

    with pytest.raises(ValueError, match="train_indices"):
        thor.training.IndexedNamedBatchLoader(
            dataset_path=dataset_path,
            layout=layout,
            train_indices=np.empty((0,), dtype=np.int64),
            validate_indices=np.empty((0,), dtype=np.int64),
            batch_size=1,
            randomize_train=False,
        )


def test_indexed_local_named_loader_supports_explicit_test_indices(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=10, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[0, 1],
        validate_indices=[2],
        test_indices=[4, 3],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.has_explicit_test_split()
    test_batch = loader.copy_next_batch("test")
    np.testing.assert_array_equal(
        test_batch["seasonality_inputs"],
        np.asarray([[41.0, 42.0], [31.0, 32.0]], dtype=np.float32),
    )


def test_indexed_local_named_loader_rejects_out_of_range_indices(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=10, storage_mode="indexed"
    )
    writer.write_indexed_example(_example(0))
    writer.close()

    with pytest.raises(RuntimeError, match="outside dataset row count"):
        thor.training.IndexedNamedBatchLoader(
            dataset_path,
            layout,
            train_indices=[0, 1],
            validate_indices=[0],
            batch_size=2,
            randomize_train=False,
        )



def test_indexed_local_named_loader_randomized_train_seed_is_deterministic(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    first = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[0, 1, 2, 3, 4],
        validate_indices=[0],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=True,
        random_seed=12345,
    )
    second = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[0, 1, 2, 3, 4],
        validate_indices=[0],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=True,
        random_seed=12345,
    )

    for _ in range(4):
        first_batch = first.copy_next_batch("train")
        second_batch = second.copy_next_batch("train")
        np.testing.assert_array_equal(first_batch["seasonality_inputs"], second_batch["seasonality_inputs"])
        np.testing.assert_array_equal(first_batch["daily_target"], second_batch["daily_target"])


def test_indexed_local_named_loader_validate_and_test_are_sequential_and_wrap(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[4, 3],
        validate_indices=[1, 3, 4],
        test_indices=[2, 0, 1],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=True,
        random_seed=9876,
    )

    validate0 = loader.copy_next_batch("validate")
    validate1 = loader.copy_next_batch("validate")
    test0 = loader.copy_next_batch("test")
    test1 = loader.copy_next_batch("test")

    np.testing.assert_array_equal(
        validate0["seasonality_inputs"],
        np.asarray([[11.0, 12.0], [31.0, 32.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        validate1["seasonality_inputs"],
        np.asarray([[41.0, 42.0], [11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        test0["seasonality_inputs"],
        np.asarray([[21.0, 22.0], [1.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        test1["seasonality_inputs"],
        np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32),
    )


def test_indexed_local_named_loader_two_fold_loaders_share_one_dataset(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    fold_a = thor.training.IndexedNamedBatchLoader(
        dataset_path, layout, train_indices=[0, 2, 4], validate_indices=[1], batch_size=2, randomize_train=False
    )
    fold_b = thor.training.IndexedNamedBatchLoader(
        dataset_path, layout, train_indices=[1, 3], validate_indices=[0, 4], batch_size=2, randomize_train=False
    )

    batch_a = fold_a.copy_next_batch("train")
    batch_b = fold_b.copy_next_batch("train")

    np.testing.assert_array_equal(
        batch_a["seasonality_inputs"],
        np.asarray([[1.0, 2.0], [21.0, 22.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch_b["seasonality_inputs"],
        np.asarray([[11.0, 12.0], [31.0, 32.0]], dtype=np.float32),
    )


def test_indexed_local_named_loader_rejects_requested_layout_mismatch(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    writer.write_indexed_example(_example(0))
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
        thor.training.IndexedNamedBatchLoader(
            dataset_path,
            wrong_layout,
            train_indices=[0],
            validate_indices=[0],
            batch_size=1,
            randomize_train=False,
        )


def test_local_named_writer_supports_indexed_storage_mode_manifest(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    assert writer.get_storage_mode() == "indexed"
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["storage_mode"] == "indexed"
    assert manifest["num_examples"] == 5
    assert [shard["global_start"] for shard in manifest["shards"]] == [0, 2, 4]
    assert [shard["num_examples"] for shard in manifest["shards"]] == [2, 2, 1]


def test_indexed_local_named_loader_rejects_split_storage_mode_dataset(tmp_path):
    dataset_path = tmp_path / "split_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(dataset_path, layout, examples_per_shard=10)
    writer.write_example(_example(0), split="train")
    writer.close()

    with pytest.raises(RuntimeError, match="indexed"):
        thor.training.IndexedNamedBatchLoader(
            dataset_path,
            layout,
            train_indices=[0],
            validate_indices=[0],
            batch_size=1,
            randomize_train=False,
        )


def test_local_named_batch_loader_rejects_indexed_storage_mode_dataset(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=10, storage_mode="indexed"
    )
    writer.write_indexed_example(_example(0))
    writer.close()

    with pytest.raises(RuntimeError, match="split local named dataset"):
        thor.training.LocalNamedBatchLoader(dataset_path, layout, batch_size=1)


def _chunk(start: int, count: int) -> dict[str, np.ndarray]:
    examples = [_example(i) for i in range(start, start + count)]
    return {
        name: np.stack([example[name] for example in examples], axis=0)
        for name in examples[0]
    }


def test_local_named_writer_chunked_indexed_examples_with_expected_count_and_preallocation(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path,
        layout,
        examples_per_shard=2,
        storage_mode="indexed",
        expected_num_examples=5,
        preallocate=True,
    )
    assert writer.get_expected_num_examples() == 5
    assert writer.get_preallocate()
    writer.write_indexed_examples(_chunk(0, 3))
    writer.write_indexed_examples(_chunk(3, 2))
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["storage_mode"] == "indexed"
    assert manifest["expected_num_examples"] == 5
    assert manifest["preallocated"] is True
    assert [shard["global_start"] for shard in manifest["shards"]] == [0, 2, 4]
    assert [shard["capacity_examples"] for shard in manifest["shards"]] == [2, 2, 1]
    assert [shard["num_examples"] for shard in manifest["shards"]] == [2, 2, 1]

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[4, 2, 0],
        validate_indices=[1, 3],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )
    batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(
        batch["seasonality_inputs"],
        np.asarray([[41.0, 42.0], [21.0, 22.0]], dtype=np.float32),
    )


def test_local_named_writer_preallocate_requires_expected_num_examples(tmp_path):
    with pytest.raises(RuntimeError, match="expected_num_examples"):
        thor.training.LocalNamedExampleDatasetWriter(
            tmp_path / "indexed_named_dataset",
            _layout(),
            examples_per_shard=10,
            storage_mode="indexed",
            preallocate=True,
        )


def test_local_named_writer_expected_num_examples_enforced_on_close(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(
        tmp_path / "indexed_named_dataset",
        _layout(),
        examples_per_shard=10,
        storage_mode="indexed",
        expected_num_examples=2,
    )
    writer.write_indexed_examples(_chunk(0, 1))
    with pytest.raises(RuntimeError, match="expected_num_examples"):
        writer.close()


def test_local_named_writer_rejects_chunk_shape_mismatch(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(
        tmp_path / "indexed_named_dataset",
        _layout(),
        examples_per_shard=10,
        storage_mode="indexed",
    )
    chunk = _chunk(0, 2)
    chunk["daily_target"] = np.asarray([1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="shape"):
        writer.write_indexed_examples(chunk)


def _windowed_layout() -> thor.training.LocalNamedExampleLayout:
    return thor.training.LocalNamedExampleLayout(
        tensors={"dense": [1]},
        windowed_tensors={
            "history": thor.training.WindowedTensorLayout(
                shape=[3, 1],
                data_type=thor.DataType.fp32,
                key_type=thor.DataType.uint64,
                index_type=thor.DataType.int32,
                pad=thor.training.ConstantPad(-1.0),
                mask_name="history_mask",
            )
        },
        data_type=thor.DataType.fp32,
    )


def test_windowed_local_named_layout_exposes_python_contract():
    layout = _windowed_layout()

    assert layout.has_windowed_tensors()
    assert layout.get_tensor_names() == ["dense", "history", "history_mask"]
    assert layout.get_tensor_shapes() == {
        "dense": [1],
        "history": [3, 1],
        "history_mask": [3],
    }
    assert layout.get_record_size_bytes() == 16
    assert layout.get_windowed_tensor_specs() == {
        "history": {
            "shape": [3, 1],
            "data_type": thor.DataType.fp32,
            "key_type": thor.DataType.uint64,
            "index_type": thor.DataType.int32,
            "pad_value": -1.0,
            "mask_name": "history_mask",
            "reference_offset_bytes": 4,
            "reference_num_bytes": 12,
            "num_bytes": 12,
            "source_filename": None,
            "source_num_bytes": 0,
            "source_sequences": [],
        }
    }


def test_indexed_local_named_windowed_tensor_round_trip_from_python(tmp_path):
    dataset_path = tmp_path / "windowed_named_dataset"
    layout = _windowed_layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=10,
        storage_mode="indexed",
    )
    writer.write_windowed_tensor_source(
        tensor_name="history",
        key=7,
        start_index=10,
        values=np.asarray([[10.0], [11.0], [12.0], [13.0]], dtype=np.float32),
    )
    writer.write_indexed_examples(
        {
            "dense": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "history": thor.training.WindowedTensorChunk(
                key=np.asarray([7, 7, 7], dtype=np.uint64),
                start=np.asarray([10, 8, 12], dtype=np.int32),
            ),
        }
    )
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    history_manifest = manifest["windowed_tensors"]["history"]
    assert history_manifest["source_storage"]["file"] == "windowed_tensor_sources/windowed_tensor_000000.bin"
    assert history_manifest["source_storage"]["sequences"] == [
        {
            "key_hex": "0700000000000000",
            "start_index": 10,
            "end_index_exclusive": 14,
            "offset_bytes": 0,
            "num_steps": 4,
            "num_bytes": 16,
        }
    ]

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=[0, 1, 2],
        validate_indices=[],
        test_indices=[],
        batch_size=3,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.get_tensor_names() == ["dense", "history", "history_mask"]
    assert loader.get_tensor_shapes() == layout.get_tensor_shapes()

    batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(batch["dense"], np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        batch["history"],
        np.asarray(
            [
                [[10.0], [11.0], [12.0]],
                [[-1.0], [-1.0], [10.0]],
                [[12.0], [13.0], [-1.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch["history_mask"],
        np.asarray(
            [
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        ),
    )

    stats = loader.get_stats("train")
    assert stats["windowed_source_read_calls"] >= 3
    assert stats["windowed_source_read_bytes"] >= 24


def test_windowed_indexed_writer_accepts_single_example_reference_from_python(tmp_path):
    dataset_path = tmp_path / "windowed_single_example_dataset"
    layout = _windowed_layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=10,
        storage_mode="indexed",
    )
    writer.write_windowed_tensor_source(
        "history",
        key=99,
        start_index=0,
        values=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
    )
    writer.write_indexed_example(
        {
            "dense": np.asarray([123.0], dtype=np.float32),
            "history": thor.training.WindowedTensorChunk(key=99, start=0),
        }
    )
    writer.close()

    loader = thor.training.IndexedNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[0],
        validate_indices=[],
        test_indices=[],
        batch_size=1,
        batch_queue_depth=2,
        randomize_train=False,
    )

    batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(batch["dense"], np.asarray([[123.0]], dtype=np.float32))
    np.testing.assert_array_equal(batch["history"], np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32))
    np.testing.assert_array_equal(batch["history_mask"], np.asarray([[1, 1, 1]], dtype=np.uint8))


def test_windowed_writer_rejects_wrong_reference_array_dtype_from_python(tmp_path):
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path=tmp_path / "windowed_bad_refs",
        layout=_windowed_layout(),
        examples_per_shard=10,
        storage_mode="indexed",
    )

    with pytest.raises(TypeError, match="numpy.uint64"):
        writer.write_indexed_examples(
            {
                "dense": np.asarray([[1.0]], dtype=np.float32),
                "history": thor.training.WindowedTensorChunk(
                    key=np.asarray([7], dtype=np.int64),
                    start=np.asarray([0], dtype=np.int32),
                ),
            }
        )


def test_dataset_split_manifests_bind_folds_to_one_dataset_and_round_trip(tmp_path):
    dataset_path = tmp_path / "manifest_indexed_named_dataset"
    layout = _layout()
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=3, storage_mode="indexed"
    )
    for i in range(10):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    groups = [i // 2 for i in range(10)]
    folds = thor.data.StratifiedSplitter.k_fold_manifests(
        dataset=dataset,
        strata=[float(group) for group in groups],
        groups=groups,
        mode="quantile",
        num_bins=5,
        k=5,
        seed=23,
    )

    assert len(folds) == 5
    assert all(manifest.dataset_id == dataset.id for manifest in folds)
    assert all(manifest.num_examples == dataset.num_examples for manifest in folds)
    assert all(manifest.test_aliases_validate for manifest in folds)
    assert all(not manifest.has_explicit_test_split for manifest in folds)

    validation_sets = [set(manifest.validate.indices) for manifest in folds]
    assert set().union(*validation_sets) == set(range(10))
    assert sum(len(indices) for indices in validation_sets) == 10
    for indices in validation_sets:
        assert len(indices) == 2
        only_group = {groups[index] for index in indices}
        assert len(only_group) == 1

    manifest_path = tmp_path / "fold_0.json"
    folds[0].save(manifest_path)
    loaded = thor.data.DatasetSplitManifest.load(manifest_path)
    assert loaded == folds[0]
    assert loaded.test_aliases_validate
    loaded.validate_against(dataset)

    policy = thor.data.BatchPolicy(batch_size=2, randomize_train=False)
    loader = thor.training.IndexedNamedBatchLoader(
        dataset=dataset,
        splits=loaded,
        batching=policy,
        batch_queue_depth=2,
    )
    expected_rows = list(loaded.train.indices[:2])
    batch = loader.copy_next_batch("train")
    np.testing.assert_array_equal(
        batch["seasonality_inputs"],
        np.asarray([[float(row * 10 + 1), float(row * 10 + 2)] for row in expected_rows], dtype=np.float32),
    )
    assert loader.get_split_manifest() == loaded


def test_dataset_split_manifest_rejects_wrong_dataset_and_invalid_membership(tmp_path):
    layout = _layout()

    def write_dataset(path, count):
        writer = thor.training.LocalNamedExampleDatasetWriter(
            path, layout, examples_per_shard=2, storage_mode="indexed"
        )
        for i in range(count):
            writer.write_indexed_example(_example(i))
        writer.close()
        return thor.data.LocalNamedDataset.open(path)

    dataset_a = write_dataset(tmp_path / "dataset_a", 4)
    dataset_b = write_dataset(tmp_path / "dataset_b", 4)
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset_a,
        train_indices=[0, 1, 2],
        validate_indices=[3],
    )

    with pytest.raises(RuntimeError, match="different dataset identity"):
        manifest.validate_against(dataset_b)

    with pytest.raises(RuntimeError, match="outside dataset row count"):
        thor.data.DatasetSplitManifest(
            dataset=dataset_a,
            train_indices=[0, 4],
            validate_indices=[1],
        )

    with pytest.raises(RuntimeError, match="duplicate row index"):
        thor.data.DatasetSplitManifest(
            dataset=dataset_a,
            train_indices=[0, 0],
            validate_indices=[1],
        )

    explicit = thor.data.DatasetSplitManifest(
        dataset=dataset_a,
        train_indices=[0, 1],
        validate_indices=[2],
        test_indices=[3],
    )
    assert explicit.has_explicit_test_split
    assert not explicit.test_aliases_validate
    assert explicit.test.indices == [3]


def test_training_data_opens_independent_named_batch_sessions(tmp_path):
    dataset_path = tmp_path / "training_data_sessions"
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, _layout(), examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(6):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    splits = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=[0, 1, 2, 3],
        validate_indices=[4, 5],
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=False),
        dataset_name="shared_examples",
    )

    first = data.open_session(max_in_flight_batches=2)
    second = data.open_session(max_in_flight_batches=2)
    assert first is not second
    assert isinstance(first, thor.data.IndexedNamedBatchSession)
    assert isinstance(first, thor.data.IndexedNamedBatchLoader)
    assert isinstance(first, thor.training.IndexedNamedBatchSession)
    assert isinstance(first, thor.training.IndexedNamedBatchLoader)
    assert isinstance(first, thor.training.IndexedNamedBatchLoader)
    assert first.get_dataset_name() == "shared_examples"
    assert second.get_dataset_name() == "shared_examples"

    first_batch = first.copy_next_batch("train")
    second_batch = second.copy_next_batch("train")
    np.testing.assert_array_equal(
        first_batch["seasonality_inputs"],
        np.asarray([[1.0, 2.0], [11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        second_batch["seasonality_inputs"],
        first_batch["seasonality_inputs"],
    )

    next_first = first.copy_next_batch("train")
    repeated_second = second.copy_next_batch("train")
    np.testing.assert_array_equal(
        next_first["seasonality_inputs"],
        np.asarray([[21.0, 22.0], [31.0, 32.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(repeated_second["seasonality_inputs"], next_first["seasonality_inputs"])


def test_training_data_owns_device_access_policy(tmp_path):
    dataset_path = tmp_path / "training_data_access_policy"
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, _layout(), examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(3):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    splits = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=[0, 1],
        validate_indices=[2],
    )
    batching = thor.data.BatchPolicy(batch_size=1, randomize_train=False)

    default_data = thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=batching,
    )
    strict_data = thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=batching,
        device_storage="strict",
    )

    assert default_data.device_storage == thor.data.DeviceDatasetStorage.BEST_EFFORT
    assert strict_data.device_storage == thor.data.DeviceDatasetStorage.STRICT
    assert not hasattr(thor.training.TrainerFitOptions(), "device_dataset_storage")

    policy = thor.data.DatasetAccessPolicy()
    assert policy.device_storage == thor.data.DeviceDatasetStorage.BEST_EFFORT
    policy.device_storage = "off"
    assert policy.device_storage == thor.data.DeviceDatasetStorage.OFF


def test_training_data_session_cancellation_is_local(tmp_path):
    dataset_path = tmp_path / "training_data_cancel"
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, _layout(), examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(4):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=[0, 1, 2],
            validate_indices=[3],
        ),
        batching=thor.data.BatchPolicy(batch_size=1, randomize_train=False),
    )
    cancelled = data.open_session(max_in_flight_batches=1)
    survivor = data.open_session(max_in_flight_batches=1)
    cancelled.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        cancelled.copy_next_batch("train")
    np.testing.assert_array_equal(
        survivor.copy_next_batch("train")["seasonality_inputs"],
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="max_in_flight_batches"):
        data.open_session(max_in_flight_batches=0)


def _write_binding_dataset(tmp_path):
    dataset_path = tmp_path / "dataset_input_bindings"
    layout = _layout()
    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(4):
        writer.write_indexed_example(_example(i))
    writer.close()
    dataset = thor.data.LocalNamedDataset.open(dataset_path)
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=[0, 1, 2],
            validate_indices=[3],
        ),
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=False),
    )
    return dataset, data


def test_dataset_input_bindings_support_explicit_names_and_exact_name_autobinding(tmp_path):
    dataset, data = _write_binding_dataset(tmp_path)

    explicit_network = thor.Network("explicit_dataset_bindings")
    explicit_input = thor.layers.NetworkInput(
        explicit_network,
        "model_seasonality",
        [2],
        thor.DataType.fp32,
    )
    explicit = thor.training.DatasetInputBindings()
    explicit.bind(explicit_input, dataset.field("seasonality_inputs"))
    assert len(explicit) == 1
    assert not explicit.empty
    trainer = thor.training.Trainer(
        network=explicit_network,
        data=data,
        input_bindings=explicit,
    )
    assert trainer is not None

    exact_network = thor.Network("exact_dataset_bindings")
    thor.layers.NetworkInput(
        exact_network,
        "seasonality_inputs",
        [2],
        thor.DataType.fp32,
    )
    exact = thor.training.DatasetInputBindings.by_exact_name(
        network=exact_network,
        dataset=dataset,
    )
    assert len(exact) == 1
    assert thor.training.Trainer(
        network=exact_network,
        data=data,
        input_bindings=exact,
    ) is not None


def test_dataset_input_bindings_reject_hidden_conversion_missing_and_duplicate_bindings(tmp_path):
    dataset, data = _write_binding_dataset(tmp_path)

    dtype_network = thor.Network("dtype_mismatch_dataset_bindings")
    dtype_input = thor.layers.NetworkInput(
        dtype_network,
        "seasonality_inputs",
        [2],
        thor.DataType.fp16,
    )
    dtype_bindings = thor.training.DatasetInputBindings()
    dtype_bindings.bind(dtype_input, dataset.field("seasonality_inputs"))
    with pytest.raises(RuntimeError, match="dtype mismatch|TypeConversion"):
        thor.training.Trainer(
            network=dtype_network,
            data=data,
            input_bindings=dtype_bindings,
        )

    missing_network = thor.Network("missing_dataset_binding")
    first = thor.layers.NetworkInput(
        missing_network,
        "seasonality_inputs",
        [2],
        thor.DataType.fp32,
    )
    thor.layers.NetworkInput(
        missing_network,
        "daily_target",
        [2],
        thor.DataType.fp32,
    )
    missing_bindings = thor.training.DatasetInputBindings()
    missing_bindings.bind(first, dataset.field("seasonality_inputs"))
    with pytest.raises(RuntimeError, match="missing required external NetworkInput"):
        thor.training.Trainer(
            network=missing_network,
            data=data,
            input_bindings=missing_bindings,
        )

    duplicate_network = thor.Network("duplicate_dataset_binding")
    duplicate_input = thor.layers.NetworkInput(
        duplicate_network,
        "seasonality_inputs",
        [2],
        thor.DataType.fp32,
    )
    duplicate_bindings = thor.training.DatasetInputBindings()
    duplicate_bindings.bind(duplicate_input, dataset.field("seasonality_inputs"))
    with pytest.raises(RuntimeError, match="duplicate binding"):
        duplicate_bindings.bind(duplicate_input, dataset.field("daily_target"))

    renamed_network = thor.Network("strict_exact_name_binding")
    thor.layers.NetworkInput(
        renamed_network,
        "renamed_seasonality",
        [2],
        thor.DataType.fp32,
    )
    with pytest.raises(RuntimeError, match="could not find dataset field"):
        thor.training.DatasetInputBindings.by_exact_name(
            network=renamed_network,
            dataset=dataset,
        )


def test_dataset_input_bindings_allow_graph_level_type_conversion(tmp_path):
    dataset, data = _write_binding_dataset(tmp_path)
    network = thor.Network("graph_level_type_conversion")
    raw = thor.layers.NetworkInput(
        network,
        "seasonality_inputs",
        [2],
        thor.DataType.fp32,
    )
    converted = thor.layers.TypeConverter(
        network,
        raw.get_feature_output(),
        thor.DataType.fp16,
    )
    assert converted is not None

    bindings = thor.training.DatasetInputBindings.by_exact_name(
        network=network,
        dataset=dataset,
    )
    assert thor.training.Trainer(
        network=network,
        data=data,
        input_bindings=bindings,
    ) is not None


def test_phase_training_binds_only_active_dataset_field_subset(tmp_path):
    dataset, data = _write_binding_dataset(tmp_path)

    daily = thor.Network("phase_subset_daily")
    daily_features = thor.layers.NetworkInput(daily, "seasonality_inputs", [2], thor.DataType.fp32)
    daily_labels = thor.layers.NetworkInput(daily, "daily_target", [2], thor.DataType.fp32)
    daily_prediction = thor.layers.FullyConnected(
        daily,
        daily_features.get_feature_output(),
        2,
        True,
        activation=None,
    ).get_feature_output()
    daily_loss = thor.losses.MSE(
        daily,
        daily_prediction,
        daily_labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(daily, "daily_loss", daily_loss.get_loss(), thor.DataType.fp32)

    aggregate = thor.Network("phase_subset_aggregate")
    aggregate_features = thor.layers.NetworkInput(aggregate, "monotone_inputs", [3], thor.DataType.fp32)
    aggregate_labels = thor.layers.NetworkInput(aggregate, "daily_target", [2], thor.DataType.fp32)
    aggregate_prediction = thor.layers.FullyConnected(
        aggregate,
        aggregate_features.get_feature_output(),
        2,
        True,
        activation=None,
    ).get_feature_output()
    aggregate_loss = thor.losses.MSE(
        aggregate,
        aggregate_prediction,
        aggregate_labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(aggregate, "aggregate_loss", aggregate_loss.get_loss(), thor.DataType.fp32)

    daily_phase = thor.training.TrainingPhase("daily", network=daily, enabled=True)
    aggregate_phase = thor.training.TrainingPhase("aggregate", network=aggregate, enabled=False)
    program = thor.training.TrainingProgram([
        thor.training.TrainingStep(
            "daily_then_aggregate",
            phases=[daily_phase, aggregate_phase],
            optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        )
    ])
    trainer = thor.training.Trainer(
        data=data,
        training_program=program,
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        stats_color="never",
    )

    # The dataset contains fields not consumed by the active daily phase. Only
    # seasonality_inputs and daily_target are validated and materialized.
    trainer.fit(epochs=1, max_training_batches_per_epoch=1)

    # Enabling the sibling phase recomposes the model and extends the consumed
    # subset to monotone_inputs without changing the Trainer.
    aggregate_phase.enable()
    trainer.fit(epochs=1, max_training_batches_per_epoch=1)
