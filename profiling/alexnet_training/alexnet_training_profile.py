#!/usr/bin/env python3
"""Reproducible AlexNet Trainer workload for Nsight Systems timeline analysis.

The case intentionally keeps dataset preparation outside the capture and uses
host-resident TrainingData (`device_storage="off"`) so the profile exposes the
NetworkInput HtoD pipeline that should overlap with current-batch compute.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import thor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "profiling/data"
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "alexnet_imagenet100"
DEFAULT_OUTPUT = REPO_ROOT / "build/profiling/alexnet_training/alexnet_trainer.nsys-rep"
DEFAULT_DATASET_ID = "clane9/imagenet-100"
IMAGE_SIZE = 224
RESIZE_SHORTER_SIDE = 256
NUM_CLASSES = 100
CACHE_VERSION = 2
DEFAULT_TRAIN_EXAMPLES = 20_000
DEFAULT_VALIDATE_EXAMPLES = 5_000


def _center_crop_resize_to_chw_fp16(image) -> np.ndarray:
    from PIL import Image  # type: ignore

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {image.size}")

    if width < height:
        new_width = RESIZE_SHORTER_SIDE
        new_height = int(round(height * RESIZE_SHORTER_SIDE / width))
    else:
        new_height = RESIZE_SHORTER_SIDE
        new_width = int(round(width * RESIZE_SHORTER_SIDE / height))

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((new_width, new_height), resampling)
    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array, dtype=np.float16)


def _cache_key(dataset_id: str, train_examples: int, validate_examples: int) -> str:
    safe_dataset = dataset_id.replace("/", "_").replace(":", "_")
    return f"v{CACHE_VERSION}_{safe_dataset}_{IMAGE_SIZE}_fp16_train{train_examples}_validate{validate_examples}"


def _load_or_build_imagenet_subset(
    *,
    dataset_id: str,
    cache_root: Path,
    train_examples: int,
    validate_examples: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Download requested train/validation prefixes and cache preprocessed NumPy tensors."""

    total_examples = train_examples + validate_examples
    subset_dir = cache_root / _cache_key(dataset_id, train_examples, validate_examples)
    examples_path = subset_dir / "examples.npy"
    labels_path = subset_dir / "labels.npy"
    manifest_path = subset_dir / "manifest.json"

    if rebuild and subset_dir.exists():
        shutil.rmtree(subset_dir)

    expected_manifest = {
        "cache_version": CACHE_VERSION,
        "dataset_id": dataset_id,
        "image_size": IMAGE_SIZE,
        "resize_shorter_side": RESIZE_SHORTER_SIDE,
        "num_classes": NUM_CLASSES,
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "train_split": "train",
        "validate_split": "validation",
    }
    if examples_path.exists() and labels_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in expected_manifest.items()):
            print(f"alexnet profile: using cached ImageNet subset at {subset_dir}", flush=True)
            examples = np.load(examples_path)
            labels = np.load(labels_path)
            return np.ascontiguousarray(examples), np.ascontiguousarray(labels)

    try:
        from datasets import load_dataset  # type: ignore
        from PIL import Image  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AlexNet profiling requires the optional Python packages 'datasets' and 'Pillow'. "
            "Install them in the Thor venv with: ./.venv/bin/pip install datasets Pillow"
        ) from exc

    subset_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = cache_root / "hf_datasets"
    examples = np.empty((total_examples, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float16)
    labels = np.zeros((total_examples, NUM_CLASSES), dtype=np.float16)

    def materialize_split(split_name: str, count: int, destination_offset: int) -> None:
        if count == 0:
            return
        print(
            f"alexnet profile: streaming {count} examples from {dataset_id} split={split_name!r}; "
            "only the requested prefix is materialized locally",
            flush=True,
        )
        stream = load_dataset(
            dataset_id,
            split=split_name,
            streaming=True,
            cache_dir=str(hf_cache),
        )

        label_feature = stream.features.get("label") if stream.features is not None else None
        label_names = list(getattr(label_feature, "names", [])) if label_feature is not None else []
        if label_names and len(label_names) != NUM_CLASSES:
            raise RuntimeError(
                f"expected {NUM_CLASSES} ImageNet-100 classes in split {split_name!r}, got {len(label_names)}"
            )

        iterator = iter(stream)
        for split_index in range(count):
            try:
                row = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"dataset {dataset_id!r} split {split_name!r} ended after {split_index} examples; "
                    f"requested {count}"
                ) from exc
            label = int(row["label"])
            if label < 0 or label >= NUM_CLASSES:
                raise RuntimeError(
                    f"source split {split_name!r} example {split_index} has label {label}, "
                    f"outside [0, {NUM_CLASSES})"
                )
            destination_index = destination_offset + split_index
            examples[destination_index] = _center_crop_resize_to_chw_fp16(row["image"])
            labels[destination_index, label] = np.float16(1.0)
            if (split_index + 1) % 128 == 0 or split_index + 1 == count:
                print(
                    f"alexnet profile: preprocessed split={split_name} {split_index + 1}/{count}",
                    flush=True,
                )

    materialize_split("train", train_examples, 0)
    materialize_split("validation", validate_examples, train_examples)

    np.save(examples_path, examples, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True))
    print(f"alexnet profile: cached preprocessed subset at {subset_dir}", flush=True)
    return examples, labels


def _make_training_data(
    examples: np.ndarray,
    labels: np.ndarray,
    *,
    train_examples: int,
    validate_examples: int,
    batch_size: int,
) -> thor.data.TrainingData:
    dataset = thor.data.NumpyDataset({"examples": examples, "labels": labels})
    train_indices = np.arange(train_examples, dtype=np.int64)
    validate_indices = np.arange(train_examples, train_examples + validate_examples, dtype=np.int64)
    splits = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=train_indices,
        validate_indices=validate_indices,
    )
    batching = thor.data.BatchPolicy(
        batch_size=batch_size,
        randomize_train=False,
    )
    return thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=batching,
        dataset_name="alexnet_profile_imagenet100_subset",
        # Deliberate: this case exists partly to verify that next-batch HtoD
        # remains overlapped with current-batch compute.
        device_storage="off",
    )


def _build_alexnet() -> thor.Network:
    network = thor.Network("profiling_alexnet")
    examples = thor.layers.NetworkInput(
        network, "examples", [3, IMAGE_SIZE, IMAGE_SIZE], thor.DataType.fp16
    )
    labels = thor.layers.NetworkInput(network, "labels", [NUM_CLASSES], thor.DataType.fp16)

    x = examples.get_feature_output()
    # Keep the legacy AlexNet geometry explicit. Convolution2d now takes a
    # single padding=(top, bottom, left, right) argument instead of separate
    # vertical/horizontal padding positional arguments. Use keywords here so
    # future constructor additions cannot silently reinterpret this profile.
    x = thor.layers.Convolution2d(
        network,
        x,
        64,
        11,
        11,
        vertical_stride=4,
        horizontal_stride=4,
        padding=(2, 2, 2, 2),
        has_bias=True,
        activation=thor.activations.Relu(),
    ).get_feature_output()
    x = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2).get_feature_output()
    x = thor.layers.Convolution2d(
        network,
        x,
        192,
        5,
        5,
        vertical_stride=1,
        horizontal_stride=1,
        padding=(2, 2, 2, 2),
        has_bias=True,
        activation=thor.activations.Relu(),
    ).get_feature_output()
    x = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2).get_feature_output()
    x = thor.layers.Convolution2d(
        network,
        x,
        384,
        3,
        3,
        vertical_stride=1,
        horizontal_stride=1,
        padding=(1, 1, 1, 1),
        has_bias=True,
        activation=thor.activations.Relu(),
    ).get_feature_output()
    x = thor.layers.Convolution2d(
        network,
        x,
        256,
        3,
        3,
        vertical_stride=1,
        horizontal_stride=1,
        padding=(1, 1, 1, 1),
        has_bias=True,
        activation=thor.activations.Relu(),
    ).get_feature_output()
    x = thor.layers.Convolution2d(
        network,
        x,
        256,
        3,
        3,
        vertical_stride=1,
        horizontal_stride=1,
        padding=(1, 1, 1, 1),
        has_bias=True,
        activation=thor.activations.Relu(),
    ).get_feature_output()
    x = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2).get_feature_output()
    x = thor.layers.Flatten(network, x, 1).get_feature_output()
    x = thor.layers.FullyConnected(
        network, x, 4096, True, activation=thor.activations.Relu()
    ).get_feature_output()
    x = thor.layers.DropOut(network, x, 0.5).get_feature_output()
    x = thor.layers.FullyConnected(
        network, x, 4096, True, activation=thor.activations.Relu()
    ).get_feature_output()
    x = thor.layers.DropOut(network, x, 0.5).get_feature_output()
    logits = thor.layers.FullyConnected(network, x, NUM_CLASSES, True, activation=None)

    loss = thor.losses.CategoricalCrossEntropy(
        network,
        logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), thor.DataType.fp16)
    return network


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile steady-state AlexNet training through Thor Trainer/NativeQueuedTrainingRunner."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--validate-examples", type=int, default=DEFAULT_VALIDATE_EXAMPLES)
    parser.add_argument("--max-in-flight-batches", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--profile-epochs", type=int, default=1)
    parser.add_argument(
        "--max-training-batches-per-epoch",
        type=int,
        default=None,
        help="Optional cap for quick experiments; by default the whole cached training subset is used.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-nsight",
        action="store_true",
        help="Run the same workload without configuring an Nsight capture.",
    )
    parser.add_argument(
        "--overwrite-profile",
        action="store_true",
        help="Remove an existing --output report before starting.",
    )
    args = parser.parse_args()

    for name in (
        "batch_size",
        "train_examples",
        "max_in_flight_batches",
        "warmup_epochs",
        "profile_epochs",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.train_examples < args.batch_size:
        parser.error("--train-examples must contain at least one full batch")
    if args.validate_examples < 0:
        parser.error("--validate-examples must be >= 0")
    if 0 < args.validate_examples < args.batch_size:
        parser.error("--validate-examples must be 0 or contain at least one full batch")
    if args.max_training_batches_per_epoch is not None and args.max_training_batches_per_epoch <= 0:
        parser.error("--max-training-batches-per-epoch must be >= 1")
    return args


def main() -> int:
    args = _parse_args()
    total_epochs = args.warmup_epochs + args.profile_epochs

    # Keep profiling datasets/caches under profiling/data so they are easy to
    # reuse between runs and remain isolated from source-controlled artifacts.
    # The repository .gitignore excludes the whole profiling/data tree.
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    examples, labels = _load_or_build_imagenet_subset(
        dataset_id=args.dataset_id,
        cache_root=args.cache_dir,
        train_examples=args.train_examples,
        validate_examples=args.validate_examples,
        rebuild=args.rebuild_data,
    )
    data = _make_training_data(
        examples,
        labels,
        train_examples=args.train_examples,
        validate_examples=args.validate_examples,
        batch_size=args.batch_size,
    )

    nsight_profile = None
    if not args.no_nsight:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            if not args.overwrite_profile:
                raise RuntimeError(
                    f"profile output already exists: {args.output}; use --overwrite-profile or choose --output"
                )
            args.output.unlink()
        nsight_profile = thor.training.NsightProfile(
            captures=[
                thor.training.NsightProfileCapture(
                    phase="default",
                    start_epoch=args.warmup_epochs + 1,
                    epoch_count=args.profile_epochs,
                    output=args.output,
                )
            ]
        )

    print(
        "alexnet profile configuration:\n"
        f"  batch_size={args.batch_size}\n"
        f"  train_examples={args.train_examples}\n"
        f"  validate_examples={args.validate_examples}\n"
        f"  max_in_flight_batches={args.max_in_flight_batches}\n"
        f"  warmup_epochs={args.warmup_epochs}\n"
        f"  profile_epochs={args.profile_epochs}\n"
        f"  max_training_batches_per_epoch={args.max_training_batches_per_epoch}\n"
        f"  data_cache={args.cache_dir}\n"
        "  device_storage=off\n"
        f"  nsight={'off' if args.no_nsight else args.output}",
        flush=True,
    )

    network = _build_alexnet()
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.9)
    trainer = thor.training.Trainer(
        network,
        data=data,
        optimizer=optimizer,
        debug_synchronous=False,
        stats_interval_s=60.0,
        max_in_flight_batches=args.max_in_flight_batches,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
        nsight_profile=nsight_profile,
    )
    trainer.fit(
        epochs=total_epochs,
        max_training_batches_per_epoch=args.max_training_batches_per_epoch,
    )

    if not args.no_nsight:
        print(f"alexnet profile: report requested at {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
