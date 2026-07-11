from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import thor


def examples_per_shard(num_examples: int, num_shards: int) -> int:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return max(1, math.ceil(num_examples / num_shards))


def write_dense_indexed_dataset(
    *,
    dataset_path: Path,
    tensor_shapes: Mapping[str, Sequence[int]],
    data_type,
    chunks: Iterable[Mapping[str, np.ndarray]],
    expected_num_examples: int,
    num_shards: int,
) -> thor.data.FileDataset:
    layout = thor.data.DatasetLayout(
        tensors={
            name: thor.data.TensorLayout(shape=shape, data_type=data_type)
            for name, shape in tensor_shapes.items()
        },
    )
    writer = thor.data.DatasetWriter(
        dataset_path=dataset_path,
        layout=layout,
        examples_per_shard=examples_per_shard(expected_num_examples, num_shards),
        expected_num_examples=expected_num_examples,
        preallocate=True,
    )
    written = 0
    try:
        for chunk in chunks:
            if not chunk:
                raise ValueError("dataset chunks must not be empty")
            first = next(iter(chunk.values()))
            count = int(first.shape[0])
            if count <= 0:
                continue
            if any(int(value.shape[0]) != count for value in chunk.values()):
                raise ValueError("all tensors in an dataset chunk must have the same leading dimension")
            writer.write_indexed_examples(dict(chunk))
            written += count
        if written != expected_num_examples:
            raise RuntimeError(
                f"dataset writer wrote {written} examples; expected {expected_num_examples}"
            )
        writer.close()
    except Exception:
        # The writer destructor will make a best effort to close, but an
        # incomplete cache is never reusable by callers because its manifest is
        # not published until close succeeds.
        raise
    return thor.data.FileDataset.open(dataset_path)


def save_split_manifest(
    *,
    dataset: thor.data.FileDataset,
    path: Path,
    train_indices,
    validate_indices,
    test_indices=None,
) -> thor.data.DatasetSplitManifest:
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=train_indices,
        validate_indices=validate_indices,
        test_indices=test_indices,
    )
    manifest.save(path)
    return manifest


def open_training_data(
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    batch_size: int,
    dataset_name: str,
    randomize_train: bool = True,
    random_seed: int | None = None,
    device_storage: str = "off",
) -> thor.data.TrainingData:
    dataset = thor.data.FileDataset.open(dataset_path)
    splits = thor.data.DatasetSplitManifest.load(split_manifest_path)
    splits.validate_against(dataset)
    return thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=thor.data.BatchPolicy(
            batch_size=batch_size,
            randomize_train=randomize_train,
            random_seed=random_seed,
        ),
        dataset_name=dataset_name,
        device_storage=device_storage,
    )
