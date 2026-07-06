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

    loader = thor.training.IndexedLocalNamedBatchLoader(
        dataset_path=dataset_path,
        layout=layout,
        train_indices=np.asarray([4, 2, 0], dtype=np.int64),
        validate_indices=np.asarray([1, 3], dtype=np.int64),
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=False,
    )

    assert loader.get_dataset_name() == "indexed_local_named_examples"
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


def test_indexed_local_named_loader_exposes_stats(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.training.LocalNamedExampleDatasetWriter(
        dataset_path, layout, examples_per_shard=2, storage_mode="indexed"
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    loader = thor.training.IndexedLocalNamedBatchLoader(
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
    loader = thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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
        thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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
        thor.training.IndexedLocalNamedBatchLoader(
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

    first = thor.training.IndexedLocalNamedBatchLoader(
        dataset_path,
        layout,
        train_indices=[0, 1, 2, 3, 4],
        validate_indices=[0],
        batch_size=2,
        batch_queue_depth=2,
        randomize_train=True,
        random_seed=12345,
    )
    second = thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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

    fold_a = thor.training.IndexedLocalNamedBatchLoader(
        dataset_path, layout, train_indices=[0, 2, 4], validate_indices=[1], batch_size=2, randomize_train=False
    )
    fold_b = thor.training.IndexedLocalNamedBatchLoader(
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
        thor.training.IndexedLocalNamedBatchLoader(
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
        thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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

    loader = thor.training.IndexedLocalNamedBatchLoader(
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
