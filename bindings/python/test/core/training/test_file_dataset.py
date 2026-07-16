from __future__ import annotations

import json

import numpy as np
import pytest

import thor


_STORABLE_THOR_DTYPES = [
    thor.DataType.bool,
    thor.DataType.int8,
    thor.DataType.uint8,
    thor.DataType.int16,
    thor.DataType.uint16,
    thor.DataType.int32,
    thor.DataType.uint32,
    thor.DataType.int64,
    thor.DataType.uint64,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp32,
    thor.DataType.fp64,
]


def _layout() -> thor.data.DatasetLayout:
    return thor.data.DatasetLayout(
        tensors={
            "seasonality_inputs": thor.data.TensorLayout([2], thor.DataType.fp32),
            "monotone_inputs": thor.data.TensorLayout([3], thor.DataType.fp32),
            "daily_target": thor.data.TensorLayout([2], thor.DataType.fp32),
            "example_weight": thor.data.TensorLayout([1], thor.DataType.fp32),
        },
    )


def _example(i: int) -> dict[str, np.ndarray]:
    base = float(i * 10)
    return {
        "seasonality_inputs": np.asarray([base + 1.0, base + 2.0], dtype=np.float32),
        "monotone_inputs": np.asarray([base + 3.0, base + 4.0, base + 5.0], dtype=np.float32),
        "daily_target": np.asarray([base + 6.0, base + 7.0], dtype=np.float32),
        "example_weight": np.asarray([base + 8.0], dtype=np.float32),
    }


def test_dataset_layout_constructs_packed_record_layout():
    layout = _layout()

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


def test_dataset_writer_and_dataset_round_trip_indexed_manifest(tmp_path):
    dataset_path = tmp_path / "named_dataset"
    layout = _layout()

    writer = thor.data.DatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=2,
    )
    for i in range(4):
        writer.write_indexed_example(_example(i))

    assert writer.get_num_examples() == 4
    writer.close()
    assert writer.is_closed()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["format"] == "thor.dataset.v1"
    assert "data_type" not in manifest
    assert manifest["storage_mode"] == "indexed"
    assert manifest["record_size_bytes"] == layout.get_record_size_bytes()
    assert manifest["num_examples"] == 4
    assert "example_type_counts" not in manifest
    assert [shard["global_start"] for shard in manifest["shards"]] == [0, 2]
    assert [shard["num_examples"] for shard in manifest["shards"]] == [2, 2]
    assert all("example_type_counts" not in shard for shard in manifest["shards"])

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.num_examples == 4
    assert dataset.schema.field("seasonality_inputs").dimensions == [2]
    assert dataset.schema.field("daily_target").dimensions == [2]


def test_file_dataset_rejects_legacy_split_manifest(tmp_path):
    dataset_path = tmp_path / "legacy_split_dataset"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    writer.write_indexed_example(_example(0))
    writer.close()

    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["storage_mode"] = "split"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="legacy split dataset storage_mode='split'.*DatasetSplitManifest"):
        thor.data.FileDataset.open(dataset_path)


def test_file_dataset_rejects_manifest_without_dataset_id(tmp_path):
    dataset_path = tmp_path / "missing_dataset_id"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    writer.write_indexed_example(_example(0))
    writer.close()

    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["dataset_id"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="missing required dataset_id.*DatasetWriter"):
        thor.data.FileDataset.open(dataset_path)


def test_dataset_writer_supports_independent_field_dtypes(tmp_path):
    dataset_path = tmp_path / "mixed_dtype_dataset"
    layout = thor.data.DatasetLayout(
        tensors={
            "token": thor.data.TensorLayout([1], thor.DataType.uint8),
            "features": thor.data.TensorLayout([2], thor.DataType.fp16),
            "target": thor.data.TensorLayout([1], thor.DataType.fp64),
        }
    )
    writer = thor.data.DatasetWriter(dataset_path, layout, examples_per_shard=2)
    writer.write_indexed_examples(
        {
            "token": np.asarray([[1], [2]], dtype=np.uint8),
            "features": np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16),
            "target": np.asarray([[10.25], [20.5]], dtype=np.float64),
        }
    )
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["record_size_bytes"] == 13
    assert manifest["tensors"]["token"]["data_type"] == "uint8"
    assert manifest["tensors"]["features"]["data_type"] == "fp16"
    assert manifest["tensors"]["target"]["data_type"] == "fp64"

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.field("token").dtype == thor.DataType.uint8
    assert dataset.field("features").dtype == thor.DataType.fp16
    assert dataset.field("target").dtype == thor.DataType.fp64


def test_dataset_writer_rejects_missing_tensor(tmp_path):
    writer = thor.data.DatasetWriter(
        tmp_path / "named_dataset",
        _layout(),
        examples_per_shard=10,
    )
    bad_example = _example(0)
    del bad_example["daily_target"]

    with pytest.raises(RuntimeError, match="tensor count|missing tensor"):
        writer.write_indexed_example(bad_example)


def test_dataset_writer_rejects_shape_mismatch(tmp_path):
    writer = thor.data.DatasetWriter(
        tmp_path / "named_dataset",
        _layout(),
        examples_per_shard=10,
    )
    bad_example = _example(0)
    bad_example["daily_target"] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="shape"):
        writer.write_indexed_example(bad_example)


@pytest.mark.parametrize("data_type", _STORABLE_THOR_DTYPES)
def test_dataset_writer_accepts_every_storable_thor_dtype(data_type, tmp_path):
    numpy_dtype = thor.physical.numpy_dtypes.from_thor(data_type)
    values = np.arange(6, dtype=np.float32).astype(numpy_dtype).reshape(3, 2)
    values = np.ascontiguousarray(values)
    dataset_path = tmp_path / f"named_dataset_{data_type}"
    layout = thor.data.DatasetLayout(
        tensors={"values": thor.data.TensorLayout([2], data_type)},
    )
    writer = thor.data.DatasetWriter(
        dataset_path,
        layout,
        examples_per_shard=2,
        expected_num_examples=3,
        preallocate=True,
    )
    writer.write_indexed_examples({"values": values})
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.num_examples == 3
    assert dataset.field("values").dtype == data_type
    assert dataset.field("values").shape == [2]
    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["record_size_bytes"] == 2 * np.dtype(numpy_dtype).itemsize


def test_dataset_writer_requires_exact_dtype_and_contiguous_arrays(tmp_path):
    writer = thor.data.DatasetWriter(
        tmp_path / "named_dataset",
        _layout(),
        examples_per_shard=10,
    )
    bad_example = _example(0)
    bad_example["daily_target"] = np.asarray([1.0, 2.0], dtype=np.float64)

    with pytest.raises(TypeError, match=r"canonical numpy dtype for thor\.DataType\.fp32"):
        writer.write_indexed_example(bad_example)

    bad_example = _example(0)
    non_contiguous = np.zeros((2, 2), dtype=np.float32)[:, 0]
    assert not non_contiguous.flags.c_contiguous
    bad_example["daily_target"] = non_contiguous

    with pytest.raises(TypeError, match="C-contiguous numpy.ndarray"):
        writer.write_indexed_example(bad_example)


def test_file_dataset_layout_rejects_compute_only_tf32_storage(tmp_path):
    with pytest.raises((ValueError, RuntimeError)):
        layout = thor.data.DatasetLayout(
            tensors={"values": thor.data.TensorLayout([2], thor.DataType.tf32)},
        )
        thor.data.DatasetWriter(
            tmp_path / "tf32_dataset",
            layout,
            examples_per_shard=2,
        )


def test_dataset_writer_writes_indexed_global_ranges(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    for i in range(5):
        writer.write_indexed_example(_example(i))
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["storage_mode"] == "indexed"
    assert manifest["num_examples"] == 5
    assert [shard["global_start"] for shard in manifest["shards"]] == [0, 2, 4]
    assert [shard["num_examples"] for shard in manifest["shards"]] == [2, 2, 1]


def _chunk(start: int, count: int) -> dict[str, np.ndarray]:
    examples = [_example(i) for i in range(start, start + count)]
    return {
        name: np.stack([example[name] for example in examples], axis=0) for name in examples[0]
    }


def test_dataset_writer_chunked_indexed_examples_with_expected_count_and_preallocation(tmp_path):
    dataset_path = tmp_path / "indexed_named_dataset"
    layout = _layout()

    writer = thor.data.DatasetWriter(
        dataset_path,
        layout,
        examples_per_shard=2,
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

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.num_examples == 5
    assert dataset.schema.field("seasonality_inputs").dimensions == [2]


def test_dataset_writer_preallocate_requires_expected_num_examples(tmp_path):
    with pytest.raises(RuntimeError, match="expected_num_examples"):
        thor.data.DatasetWriter(
            tmp_path / "indexed_named_dataset",
            _layout(),
            examples_per_shard=10,
            preallocate=True,
        )


def test_dataset_writer_expected_num_examples_enforced_on_close(tmp_path):
    writer = thor.data.DatasetWriter(
        tmp_path / "indexed_named_dataset",
        _layout(),
        examples_per_shard=10,
        expected_num_examples=2,
    )
    writer.write_indexed_examples(_chunk(0, 1))
    with pytest.raises(RuntimeError, match="expected_num_examples"):
        writer.close()


def test_dataset_writer_rejects_chunk_shape_mismatch(tmp_path):
    writer = thor.data.DatasetWriter(
        tmp_path / "indexed_named_dataset",
        _layout(),
        examples_per_shard=10,
    )
    chunk = _chunk(0, 2)
    chunk["daily_target"] = np.asarray([1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="shape"):
        writer.write_indexed_examples(chunk)


def _windowed_layout() -> thor.data.DatasetLayout:
    return thor.data.DatasetLayout(
        tensors={"dense": thor.data.TensorLayout([1], thor.DataType.fp32)},
        window_sources={
            "history_source": thor.data.WindowedTensorSourceLayout(
                step_shape=[1],
                data_type=thor.DataType.fp32,
                key_type=thor.DataType.uint64,
            )
        },
        windowed_tensors={
            "history": thor.data.WindowedTensorLayout(
                shape=[3, 1],
                source="history_source",
                index_type=thor.DataType.int32,
                pad=thor.data.ConstantPad(-1.0),
                mask_name="history_mask",
            )
        },
    )


def test_windowed_dataset_layout_exposes_python_contract():
    layout = _windowed_layout()

    assert layout.has_windowed_tensors()
    assert layout.get_tensor_names() == ["dense", "history", "history_mask"]
    assert layout.get_tensor_shapes() == {
        "dense": [1],
        "history": [3, 1],
        "history_mask": [3],
    }
    assert layout.get_record_size_bytes() == 16
    assert layout.get_window_source_specs() == {
        "history_source": {
            "step_shape": [1],
            "data_type": thor.DataType.fp32,
            "key_type": thor.DataType.uint64,
            "step_num_bytes": 4,
            "source_filename": None,
            "source_num_bytes": 0,
            "source_sequences": [],
        }
    }
    assert layout.get_windowed_tensor_specs() == {
        "history": {
            "shape": [3, 1],
            "source": "history_source",
            "data_type": thor.DataType.fp32,
            "key_type": thor.DataType.uint64,
            "index_type": thor.DataType.int32,
            "pad_value": -1.0,
            "mask_name": "history_mask",
            "reference_mode": "indexed",
            "reference_offset_bytes": 4,
            "reference_num_bytes": 12,
            "num_bytes": 12,
        }
    }


def test_windowed_writer_rejects_wrong_reference_array_dtype_from_python(tmp_path):
    writer = thor.data.DatasetWriter(
        dataset_path=tmp_path / "windowed_bad_refs",
        layout=_windowed_layout(),
        examples_per_shard=10,
    )

    with pytest.raises(TypeError, match="numpy.uint64"):
        writer.write_indexed_examples(
            {
                "dense":
                    np.asarray([[1.0]], dtype=np.float32),
                "history":
                    thor.data.WindowedTensorChunk(
                        key=np.asarray([7], dtype=np.int64),
                        start=np.asarray([0], dtype=np.int32),
                    ),
            })


def test_windowed_writer_accepts_non_fp32_source_and_dense_dtypes(tmp_path):
    dataset_path = tmp_path / "windowed_uint8"
    layout = thor.data.DatasetLayout(
        tensors={"dense": thor.data.TensorLayout([1], thor.DataType.uint8)},
        window_sources={
            "history_source": thor.data.WindowedTensorSourceLayout(
                step_shape=[1],
                data_type=thor.DataType.uint8,
                key_type=thor.DataType.uint64,
            )
        },
        windowed_tensors={
            "history": thor.data.WindowedTensorLayout(
                shape=[3, 1],
                source="history_source",
                index_type=thor.DataType.int32,
                pad=thor.data.ConstantPad(0),
                mask_name="history_mask",
            )
        },
    )
    writer = thor.data.DatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=2,
    )
    writer.write_window_source(
        "history_source",
        key=7,
        start_index=0,
        values=np.arange(4, dtype=np.uint8).reshape(4, 1),
    )
    writer.write_indexed_example(
        {
            "dense": np.asarray([5], dtype=np.uint8),
            "history": thor.data.WindowedTensorChunk(key=7, start=1),
        }
    )
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.field("dense").dtype == thor.DataType.uint8
    assert dataset.field("history").dtype == thor.DataType.uint8
    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["window_sources"]["history_source"]["storage"]["num_bytes"] == 4


def test_affine_windowed_dataset_uses_compact_segments_and_zero_record_shards(tmp_path):
    dataset_path = tmp_path / "affine_window_source"
    layout = thor.data.DatasetLayout(
        tensors={},
        window_sources={
            "tokens": thor.data.WindowedTensorSourceLayout(
                step_shape=[],
                data_type=thor.DataType.uint8,
                key_type=thor.DataType.uint64,
            )
        },
        windowed_tensors={
            "examples": thor.data.WindowedTensorLayout(
                shape=[4],
                source="tokens",
                index_type=thor.DataType.int64,
                reference_mode="affine",
            ),
            "labels": thor.data.WindowedTensorLayout(
                shape=[4],
                source="tokens",
                index_type=thor.DataType.int64,
                reference_mode="affine",
            ),
        },
    )
    assert layout.get_record_size_bytes() == 0
    assert layout.get_windowed_tensor_specs()["examples"]["reference_mode"] == "affine"
    assert layout.get_windowed_tensor_specs()["examples"]["reference_num_bytes"] == 0

    writer = thor.data.DatasetWriter(
        dataset_path,
        layout,
        examples_per_shard=2,
        expected_num_examples=3,
        preallocate=True,
    )
    writer.write_window_source(
        "tokens", key=7, start_index=0, values=np.arange(12, dtype=np.uint8))
    writer.write_affine_examples(
        count=2,
        tensors={
            "examples": thor.data.AffineWindowedTensorChunk(
                key=7, base=0, stride=2, field_offset=0),
            "labels": thor.data.AffineWindowedTensorChunk(
                key=7, base=0, stride=2, field_offset=1),
        },
    )
    writer.write_affine_examples(
        count=1,
        tensors={
            "examples": thor.data.AffineWindowedTensorChunk(
                key=7, base=4, stride=2, field_offset=0),
            "labels": thor.data.AffineWindowedTensorChunk(
                key=7, base=4, stride=2, field_offset=1),
        },
    )
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert manifest["format"] == "thor.dataset.v1"
    assert manifest["record_size_bytes"] == 0
    assert manifest["shards"] == []
    segments = manifest["affine_window_reference_segments"]
    assert [segment["row_start"] for segment in segments] == [0]
    assert [segment["count"] for segment in segments] == [3]
    source_key_hex = manifest["window_sources"]["tokens"]["storage"]["sequences"][0]["key_hex"]
    assert segments[0]["references"] == {
        "examples": {
            "key_hex": source_key_hex,
            "base": 0,
            "stride": 2,
            "field_offset": 0,
        },
        "labels": {
            "key_hex": source_key_hex,
            "base": 0,
            "stride": 2,
            "field_offset": 1,
        },
    }
    assert thor.data.FileDataset.open(dataset_path).num_examples == 3


def test_windowed_dataset_rejects_retired_layout_versions_and_old_v1_shape(tmp_path):
    dataset_path = tmp_path / "retired_window_layout"
    layout = thor.data.DatasetLayout(
        tensors={},
        window_sources={
            "tokens": thor.data.WindowedTensorSourceLayout(
                step_shape=[], data_type=thor.DataType.uint8, key_type=thor.DataType.uint64)
        },
        windowed_tensors={
            "examples": thor.data.WindowedTensorLayout(
                shape=[4], source="tokens", index_type=thor.DataType.int64)
        },
    )
    writer = thor.data.DatasetWriter(dataset_path, layout, examples_per_shard=2)
    writer.write_window_source(
        "tokens", key=7, start_index=0, values=np.arange(8, dtype=np.uint8))
    writer.write_indexed_examples(
        {
            "examples": thor.data.WindowedTensorChunk(
                key=np.asarray([7], dtype=np.uint64),
                start=np.asarray([0], dtype=np.int64),
            )
        }
    )
    writer.close()

    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for retired_format in ("thor.dataset.v2", "thor.dataset.v3"):
        retired = dict(manifest)
        retired["format"] = retired_format
        manifest_path.write_text(json.dumps(retired))
        with pytest.raises(RuntimeError, match="unsupported manifest format"):
            thor.data.FileDataset.open(dataset_path)

    old_v1 = json.loads(json.dumps(manifest))
    del old_v1["windowed_tensors"]["examples"]["reference_mode"]
    manifest_path.write_text(json.dumps(old_v1))
    with pytest.raises((RuntimeError, KeyError)):
        thor.data.FileDataset.open(dataset_path)


def test_multiple_windowed_fields_share_one_physical_source(tmp_path):
    dataset_path = tmp_path / "shared_window_source"
    layout = thor.data.DatasetLayout(
        tensors={},
        window_sources={
            "tokens": thor.data.WindowedTensorSourceLayout(
                step_shape=[],
                data_type=thor.DataType.uint8,
                key_type=thor.DataType.uint64,
            )
        },
        windowed_tensors={
            "examples": thor.data.WindowedTensorLayout(
                shape=[4], source="tokens", index_type=thor.DataType.int64),
            "labels": thor.data.WindowedTensorLayout(
                shape=[4], source="tokens", index_type=thor.DataType.int64),
        },
    )
    writer = thor.data.DatasetWriter(dataset_path, layout, examples_per_shard=2)
    writer.write_window_source(
        "tokens", key=7, start_index=0, values=np.arange(8, dtype=np.uint8))
    writer.write_indexed_examples(
        {
            "examples": thor.data.WindowedTensorChunk(
                key=np.asarray([7, 7], dtype=np.uint64),
                start=np.asarray([0, 2], dtype=np.int64),
            ),
            "labels": thor.data.WindowedTensorChunk(
                key=np.asarray([7, 7], dtype=np.uint64),
                start=np.asarray([1, 3], dtype=np.int64),
            ),
        }
    )
    writer.close()

    manifest = json.loads((dataset_path / "manifest.json").read_text())
    assert list(manifest["window_sources"]) == ["tokens"]
    assert manifest["windowed_tensors"]["examples"]["source"] == "tokens"
    assert manifest["windowed_tensors"]["labels"]["source"] == "tokens"
    assert manifest["window_sources"]["tokens"]["storage"]["num_bytes"] == 8
    assert len(list((dataset_path / "window_sources").glob("*.bin"))) == 1

    dataset = thor.data.FileDataset.open(dataset_path)
    assert dataset.field("examples").dtype == thor.DataType.uint8
    assert dataset.field("labels").dtype == thor.DataType.uint8
    assert dataset.field("examples").dimensions == [4]
    assert dataset.field("labels").dimensions == [4]


def test_dataset_split_manifests_bind_folds_to_one_dataset_and_round_trip(tmp_path):
    dataset_path = tmp_path / "manifest_indexed_named_dataset"
    layout = _layout()
    writer = thor.data.DatasetWriter(
        dataset_path, layout, examples_per_shard=3)
    for i in range(10):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
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

    data = thor.data.TrainingData(
        dataset=dataset,
        splits=loaded,
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=False),
        device_storage="off",
    )
    assert data.splits == loaded
    assert data.dataset is dataset


def test_dataset_split_manifest_persists_compact_strided_ranges(tmp_path):
    dataset_path = tmp_path / "range_manifest_dataset"
    writer = thor.data.DatasetWriter(dataset_path, _layout(), examples_per_shard=10)
    writer.write_indexed_examples(_chunk(0, 10))
    writer.close()
    dataset = thor.data.FileDataset.open(dataset_path)

    train = thor.data.ExampleIndexSet.strided(start=0, count=5, stride=2)
    validate = thor.data.ExampleIndexSet.from_ranges(
        [thor.data.ExampleIndexRange(start=1, count=3, stride=2)])
    test = thor.data.ExampleIndexSet.contiguous(start=7, count=3)
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=train,
        validate_indices=validate,
        test_indices=test,
    )
    manifest_path = tmp_path / "compact_split.json"
    manifest.save(manifest_path)

    persisted = json.loads(manifest_path.read_text())
    assert persisted["partitions"]["train"] == {
        "ranges": [{"start": 0, "count": 5, "stride": 2}]
    }
    assert persisted["partitions"]["validate"] == {
        "ranges": [{"start": 1, "count": 3, "stride": 2}]
    }
    assert persisted["partitions"]["test"] == {
        "ranges": [{"start": 7, "count": 3, "stride": 1}]
    }

    loaded = thor.data.DatasetSplitManifest.load(manifest_path)
    assert loaded.train.is_range_backed
    assert loaded.train.ranges == [thor.data.ExampleIndexRange(start=0, count=5, stride=2)]
    assert len(loaded.train) == 5
    assert loaded.train[0] == 0
    assert loaded.train[-1] == 8
    assert loaded.train.indices == [0, 2, 4, 6, 8]
    assert loaded == manifest


def test_example_index_range_validates_compact_range_contract():
    with pytest.raises(ValueError, match="count"):
        thor.data.ExampleIndexRange(start=0, count=0)
    with pytest.raises(ValueError, match="stride"):
        thor.data.ExampleIndexRange(start=0, count=1, stride=0)
    with pytest.raises(RuntimeError, match="duplicate"):
        thor.data.ExampleIndexSet.from_ranges(
            [
                thor.data.ExampleIndexRange(start=0, count=3, stride=2),
                thor.data.ExampleIndexRange(start=3, count=2, stride=1),
            ]
        )


def test_dataset_split_manifest_rejects_wrong_dataset_and_invalid_membership(tmp_path):
    layout = _layout()

    def write_dataset(path, count):
        writer = thor.data.DatasetWriter(
            path,
            layout,
            examples_per_shard=2,
        )
        for i in range(count):
            writer.write_indexed_example(_example(i))
        writer.close()
        return thor.data.FileDataset.open(path)

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


def test_training_data_opens_fresh_opaque_batch_sessions(tmp_path):
    dataset_path = tmp_path / "training_data_sessions"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    for i in range(6):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=[0, 1, 2, 3],
            validate_indices=[4, 5],
        ),
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=False),
        dataset_name="shared_examples",
    )

    first = data.open_session(max_in_flight_batches=2)
    second = data.open_session(max_in_flight_batches=2)
    assert first is not second
    assert isinstance(first, thor.data.BatchSession)
    assert isinstance(second, thor.data.BatchSession)
    assert first.get_dataset_name() == "shared_examples"
    assert second.get_dataset_name() == "shared_examples"


def test_removed_indexed_session_and_split_writer_apis_are_not_exported():
    assert not hasattr(thor.training, "Loader")
    assert not hasattr(thor.data, "Loader")
    for namespace in (thor.data, thor.training):
        assert not hasattr(namespace, "IndexedNamedBatchSession")
        assert not hasattr(namespace, "IndexedNamedBatchLoader")
        assert not hasattr(namespace, "IndexedLocalNamedBatchLoader")
        assert not hasattr(namespace, "LocalNamedBatchLoader")
        assert not hasattr(namespace, "LocalBatchLoader")
        assert not hasattr(namespace, "create_sharded_raw_dataset")

    for legacy_name in (
        "LocalNamedExampleLayout",
        "LocalNamedExampleDatasetWriter",
        "LocalNamedDataset",
        "NumpyNamedDataset",
    ):
        assert not hasattr(thor.data, legacy_name)
        assert not hasattr(thor.training, legacy_name)

    for data_name in (
        "TensorLayout",
        "WindowedTensorSourceLayout",
        "WindowedTensorLayout",
        "WindowedTensorChunk",
        "DatasetLayout",
        "DatasetWriter",
        "FileDataset",
        "NumpyDataset",
    ):
        assert hasattr(thor.data, data_name)
        assert not hasattr(thor.training, data_name)

    writer_type = thor.data.DatasetWriter
    assert not hasattr(writer_type, "write_example")
    assert not hasattr(writer_type, "write_windowed_tensor_source")
    assert not hasattr(writer_type, "get_storage_mode")
    assert not hasattr(writer_type, "get_num_train_examples")
    assert not hasattr(writer_type, "get_num_validate_examples")
    assert not hasattr(writer_type, "get_num_test_examples")


def test_training_data_owns_device_access_policy(tmp_path):
    dataset_path = tmp_path / "training_data_access_policy"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    for i in range(3):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
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
    strict_windowed_only_data = thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=batching,
        device_storage="strict-windowed-only",
    )

    assert default_data.device_storage == thor.data.DeviceDatasetStorage.OFF
    assert strict_data.device_storage == thor.data.DeviceDatasetStorage.STRICT
    assert (
        strict_windowed_only_data.device_storage
        == thor.data.DeviceDatasetStorage.STRICT_WINDOWED_ONLY
    )
    assert not hasattr(thor.training.TrainerFitOptions(), "device_dataset_storage")

    policy = thor.data.DatasetAccessPolicy()
    assert policy.device_storage == thor.data.DeviceDatasetStorage.BEST_EFFORT
    policy.device_storage = "strict_windowed_only"
    assert policy.device_storage == thor.data.DeviceDatasetStorage.STRICT_WINDOWED_ONLY
    policy.device_storage = "off"
    assert policy.device_storage == thor.data.DeviceDatasetStorage.OFF


def test_training_data_session_cancellation_is_local(tmp_path):
    dataset_path = tmp_path / "training_data_cancel"
    writer = thor.data.DatasetWriter(
        dataset_path,
        _layout(),
        examples_per_shard=2,
    )
    for i in range(4):
        writer.write_indexed_example(_example(i))
    writer.close()

    dataset = thor.data.FileDataset.open(dataset_path)
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
    survivor.cancel()

    with pytest.raises(RuntimeError, match="max_in_flight_batches"):
        data.open_session(max_in_flight_batches=0)


def _write_binding_dataset(tmp_path):
    dataset_path = tmp_path / "dataset_input_bindings"
    layout = _layout()
    writer = thor.data.DatasetWriter(
        dataset_path, layout, examples_per_shard=2)
    for i in range(4):
        writer.write_indexed_example(_example(i))
    writer.close()
    dataset = thor.data.FileDataset.open(dataset_path)
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
    program = thor.training.TrainingProgram(
        [
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
