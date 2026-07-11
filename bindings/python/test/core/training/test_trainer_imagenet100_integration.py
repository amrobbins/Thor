import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import thor
from conftest import make_numpy_pair_training_data
from integration_flags import integration_flag_enabled, integration_skip_reason
from dataset_integration_data import open_training_data, save_split_manifest, write_dense_indexed_dataset

RUN_IMAGENET100_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_IMAGENET100_INTEGRATION")
RUN_IMAGENET100_CV5_ALEXNET_INTEGRATION = integration_flag_enabled(
    "THOR_RUN_TRAINING_IMAGENET100_CV5_ALEXNET_INTEGRATION")
RUN_IMAGENET100_CV5_RESNET18_INTEGRATION = integration_flag_enabled(
    "THOR_RUN_TRAINING_IMAGENET100_CV5_RESNET18_INTEGRATION")
RUN_IMAGENET100_ANY_INTEGRATION = any(
    [
        RUN_IMAGENET100_INTEGRATION,
        RUN_IMAGENET100_CV5_ALEXNET_INTEGRATION,
        RUN_IMAGENET100_CV5_RESNET18_INTEGRATION,
    ])
IMAGENET100_CACHE_DIR = Path(os.environ.get("THOR_IMAGENET100_CACHE_DIR", "/tmp/thor_imagenet100_training"))
IMAGENET100_DATASET_ID = os.environ.get("THOR_IMAGENET100_DATASET_ID", "clane9/imagenet-100")
IMAGENET100_IMAGE_SIZE = int(os.environ.get("THOR_IMAGENET100_IMAGE_SIZE", "224"))
IMAGENET100_RESIZE_SHORTER_SIDE = int(os.environ.get("THOR_IMAGENET100_RESIZE_SHORTER_SIDE", "256"))
IMAGENET100_BATCH_SIZE = int(os.environ.get("THOR_IMAGENET100_BATCH_SIZE", "32"))
IMAGENET100_EPOCHS = int(os.environ.get("THOR_IMAGENET100_EPOCHS", "1"))
IMAGENET100_MAX_IN_FLIGHT_BATCHES = int(os.environ.get("THOR_IMAGENET100_MAX_IN_FLIGHT_BATCHES", "4"))
IMAGENET100_STATS_INTERVAL_S = float(os.environ.get("THOR_IMAGENET100_STATS_INTERVAL_S", "5.0"))
IMAGENET100_REBUILD = os.environ.get("THOR_IMAGENET100_REBUILD") == "1"
IMAGENET100_NUM_SHARDS = int(os.environ.get("THOR_IMAGENET100_NUM_SHARDS", "1"))
IMAGENET100_CV5_BATCH_SIZE = int(os.environ.get("THOR_IMAGENET100_CV5_BATCH_SIZE", str(IMAGENET100_BATCH_SIZE)))
IMAGENET100_CV5_EPOCHS = int(os.environ.get("THOR_IMAGENET100_CV5_EPOCHS", str(IMAGENET100_EPOCHS)))
IMAGENET100_CV5_MAX_IN_FLIGHT_BATCHES = int(
    os.environ.get("THOR_IMAGENET100_CV5_MAX_IN_FLIGHT_BATCHES", str(IMAGENET100_MAX_IN_FLIGHT_BATCHES)))
IMAGENET100_CV5_STATS_INTERVAL_S = float(
    os.environ.get("THOR_IMAGENET100_CV5_STATS_INTERVAL_S", str(IMAGENET100_STATS_INTERVAL_S)))
IMAGENET100_CV5_STATS_COLOR = os.environ.get("THOR_IMAGENET100_CV5_STATS_COLOR", "never").lower()
assert IMAGENET100_CV5_STATS_COLOR in {"always", "auto", "never"}
IMAGENET100_CV5_SUMMARY_LOGS_PER_SECOND = float(
    os.environ.get("THOR_IMAGENET100_CV5_SUMMARY_LOGS_PER_SECOND", "0.5"))
IMAGENET100_CV5_MAX_PARALLEL_RUNS_RAW = os.environ.get("THOR_IMAGENET100_CV5_MAX_PARALLEL_RUNS", "1")
IMAGENET100_CV5_MAX_PARALLEL_RUNS = (
    None if IMAGENET100_CV5_MAX_PARALLEL_RUNS_RAW in {None, "", "none", "None"} else
    int(IMAGENET100_CV5_MAX_PARALLEL_RUNS_RAW))
IMAGENET100_CV5_REBUILD = os.environ.get("THOR_IMAGENET100_CV5_REBUILD") == "1"
IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS = int(os.environ.get("THOR_IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS", "0"))
IMAGENET100_CV5_LEARNING_RATE = float(os.environ.get("THOR_IMAGENET100_CV5_LEARNING_RATE", "0.01"))
IMAGENET100_CV5_DECAY = float(os.environ.get("THOR_IMAGENET100_CV5_DECAY", "0.02"))
IMAGENET100_CV5_MOMENTUM = float(os.environ.get("THOR_IMAGENET100_CV5_MOMENTUM", "0.9"))
IMAGENET100_CV5_MODEL_ARTIFACTS_DIR = Path(
    os.environ.get(
        "THOR_IMAGENET100_CV5_MODEL_ARTIFACTS_DIR",
        str(Path(tempfile.gettempdir()) / "thor_imagenet100_training_runs_cv5_model_artifacts"),
    ))
# Bump whenever the indexed dataset or split-manifest cache contract changes.
IMAGENET100_MANIFEST_VERSION = 3
IMAGENET100_CV5_MANIFEST_VERSION = 2
IMAGENET100_CV5_HOLDOUT_TEST_FRACTION = 0.10
IMAGENET100_NUM_CLASSES = 100
IMAGENET100_TRAIN_EXAMPLES = 126_689
IMAGENET100_VALIDATE_EXAMPLES = 5_000

OBJECT_DETECTION_CACHE_DIR = Path(os.environ.get("THOR_OBJECT_DETECTION_CACHE_DIR", "/tmp/thor_voc_detection_training"))
OBJECT_DETECTION_DATASET_URL = os.environ.get(
    "THOR_OBJECT_DETECTION_DATASET_URL",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
)
OBJECT_DETECTION_IMAGE_SIZE = int(os.environ.get("THOR_OBJECT_DETECTION_IMAGE_SIZE", "224"))
OBJECT_DETECTION_BATCH_SIZE = int(os.environ.get("THOR_OBJECT_DETECTION_BATCH_SIZE", "16"))
OBJECT_DETECTION_EPOCHS = int(os.environ.get("THOR_OBJECT_DETECTION_EPOCHS", str(IMAGENET100_EPOCHS)))
OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES = int(
    os.environ.get("THOR_OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES", str(IMAGENET100_MAX_IN_FLIGHT_BATCHES))
)
OBJECT_DETECTION_STATS_INTERVAL_S = float(
    os.environ.get("THOR_OBJECT_DETECTION_STATS_INTERVAL_S", str(IMAGENET100_STATS_INTERVAL_S))
)
OBJECT_DETECTION_REBUILD = os.environ.get("THOR_OBJECT_DETECTION_REBUILD") == "1"
OBJECT_DETECTION_NUM_SHARDS = int(os.environ.get("THOR_OBJECT_DETECTION_NUM_SHARDS", "1"))
OBJECT_DETECTION_MANIFEST_VERSION = 2
OBJECT_DETECTION_TRAIN_EXAMPLES = int(os.environ.get("THOR_OBJECT_DETECTION_TRAIN_EXAMPLES", "1024"))
OBJECT_DETECTION_VALIDATE_EXAMPLES = int(os.environ.get("THOR_OBJECT_DETECTION_VALIDATE_EXAMPLES", "256"))
OBJECT_DETECTION_BOX_DIMS = 4
OBJECT_DETECTION_VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.imagenet100_integration,
    pytest.mark.skipif(
        not RUN_IMAGENET100_ANY_INTEGRATION,
        reason=integration_skip_reason(
            "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION",
            "THOR_RUN_TRAINING_IMAGENET100_CV5_ALEXNET_INTEGRATION",
            "THOR_RUN_TRAINING_IMAGENET100_CV5_RESNET18_INTEGRATION",
            description="heavyweight ImageNet-100 tests",
        ),
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+)")
_RUN_STATUS_RE = re.compile(
    r"INFO runs\[(?P<run>[^\]|]+)(?:\|[^\]]+)?\]:.*\bstatus=(?P<status>completed|failed|cancelled|interrupted|oom|running|starting|not_started)\b"
)


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)




class _NativeOutputTee:
    """Mirror native stdout/stderr immediately while keeping text for assertions.

    Do not use a Python pump thread here.  ``trainer.fit()`` may hold the GIL
    while native training runs, which prevents a Python reader thread from
    draining the pipe until the fit call returns.  That makes the mirrored
    trainer stats appear only at the end of the test.

    A child ``tee`` process keeps the drain/mirror path outside the Python
    interpreter, so native trainer writes are shown in real time even while the
    pybind call owns the GIL.
    """

    def __init__(self):
        self._saved_fds = {}
        self._tee_processes = []
        self._capture_paths = []

    def __enter__(self):
        _flush_native_stdio_for_capture()
        tee_exe = shutil.which("tee")
        assert tee_exe is not None, "the temporary native-output tee requires /usr/bin/tee on PATH"

        for fd in (1, 2):
            saved_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            capture_file = tempfile.NamedTemporaryFile(
                prefix=f"thor_imagenet100_fit_fd{fd}_", suffix=".log", delete=False)
            capture_path = capture_file.name
            capture_file.close()

            process = subprocess.Popen(
                [tee_exe, capture_path],
                stdin=read_fd,
                stdout=saved_fd,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            os.close(read_fd)
            os.dup2(write_fd, fd)
            os.close(write_fd)

            self._saved_fds[fd] = saved_fd
            self._tee_processes.append(process)
            self._capture_paths.append(capture_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_native_stdio_for_capture()
        finally:
            for fd, saved_fd in self._saved_fds.items():
                os.dup2(saved_fd, fd)

            for process in self._tee_processes:
                process.wait()

            for saved_fd in self._saved_fds.values():
                os.close(saved_fd)

        return False

    def text(self) -> str:
        parts = []
        for capture_path in self._capture_paths:
            path = Path(capture_path)
            try:
                parts.append(path.read_text(errors="replace"))
            finally:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return "".join(parts)


def _captured_trainer_stats(captured_text: str):
    plain_text = _ANSI_RE.sub("", captured_text)
    stats = []
    for match in _TRAINER_STATS_RE.finditer(plain_text):
        stats.append(
            {
                "phase": match.group("phase"),
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "batch": int(match.group("batch")),
                "loss": float(match.group("loss")),
            })
    return stats


def _captured_run_statuses(captured_text: str):
    plain_text = _ANSI_RE.sub("", captured_text)
    statuses = {}
    for match in _RUN_STATUS_RE.finditer(plain_text):
        statuses[match.group("run")] = match.group("status")
    return statuses


def _fit_training_runs_and_capture_text(runs, *, epochs: int, test_loader=None):
    with _NativeOutputTee() as tee:
        results = runs.fit(epochs=epochs, test_loader=test_loader)
    return results, tee.text()


def _fit_and_capture_stats(trainer, *, epochs: int):
    with _NativeOutputTee() as tee:
        trainer.fit(epochs=epochs)
    captured_text = tee.text()
    stats = _captured_trainer_stats(captured_text)

    assert stats, "trainer emitted no parseable stats; lower THOR_IMAGENET100_STATS_INTERVAL_S if this happens"
    return stats


def _stats_phase_counts(stats):
    counts = {}
    for entry in stats:
        phase = entry["phase"]
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _assert_finite_positive_losses(stats, *, model_name: str):
    losses = [entry["loss"] for entry in stats]
    phase_counts = _stats_phase_counts(stats)
    assert losses, f"{model_name}: no losses were reported; phase_counts={phase_counts}"
    for loss in losses:
        assert math.isfinite(loss), f"{model_name}: non-finite loss reported: {loss}; phase_counts={phase_counts}"
        assert loss > 0.0, f"{model_name}: non-positive loss reported: {loss}; phase_counts={phase_counts}"
    assert any(entry["phase"] == "train" for entry in stats), f"{model_name}: no train stats reported; phase_counts={phase_counts}"
    assert any(entry["phase"] == "validate" for entry in stats), (
        f"{model_name}: no validate stats reported; phase_counts={phase_counts}. "
        "This usually means the interval-throttled LineStatsReporter never printed during the short validate phase. "
        "Re-run with THOR_IMAGENET100_STATS_INTERVAL_S=0.0 to print every stats event and verify whether validate is actually running.")


def _import_imagenet_dependencies():
    try:
        from datasets import load_dataset  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The heavyweight ImageNet-100 integration tests require the optional "
            "Python packages 'datasets' and 'Pillow'. Install them in the test venv "
            "before setting THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1 "
            "or THOR_ALL_INTEGRATION_TESTS=1.") from exc
    return load_dataset, Image


def _center_crop_resize_to_chw_fp16(image, *, image_size: int, resize_shorter_side: int) -> np.ndarray:
    from PIL import Image  # type: ignore

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {image.size}")

    if width < height:
        new_width = resize_shorter_side
        new_height = int(round(height * resize_shorter_side / width))
    else:
        new_height = resize_shorter_side
        new_width = int(round(width * resize_shorter_side / height))

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((new_width, new_height), resampling)

    left = (new_width - image_size) // 2
    top = (new_height - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr, dtype=np.float16)


def _manifest_path(cache_root: Path) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_fp16_manifest.json"


def _imagenet100_dataset_path(cache_root: Path) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_indexed_v3_fp16_chw"


def _imagenet100_split_manifest_path(cache_root: Path) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_split_v3.json"


def _base_manifest(
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    num_classes: int,
    train_examples: int,
    validate_examples: int,
    label_names: list[str],
) -> dict:
    return {
        "version": IMAGENET100_MANIFEST_VERSION,
        "dataset_id": IMAGENET100_DATASET_ID,
        "image_size": IMAGENET100_IMAGE_SIZE,
        "resize_shorter_side": IMAGENET100_RESIZE_SHORTER_SIDE,
        "dtype": "fp16",
        "num_classes": num_classes,
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": validate_examples,
        "example_shape": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
        "label_shape": [num_classes],
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "label_names": list(label_names),
    }


def _read_manifest_if_valid(cache_root: Path):
    manifest_file = _manifest_path(cache_root)
    if IMAGENET100_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None
    expected = {
        "version": IMAGENET100_MANIFEST_VERSION,
        "dataset_id": IMAGENET100_DATASET_ID,
        "image_size": IMAGENET100_IMAGE_SIZE,
        "resize_shorter_side": IMAGENET100_RESIZE_SHORTER_SIDE,
        "dtype": "fp16",
        "example_shape": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    dataset_path = manifest.get("dataset_path")
    split_manifest_path = manifest.get("split_manifest_path")
    if not dataset_path or not (Path(dataset_path) / "manifest.json").exists():
        return None
    if not split_manifest_path or not Path(split_manifest_path).exists():
        return None
    if manifest.get("label_shape") != [manifest.get("num_classes")]:
        return None
    return manifest


def _imagenet100_dataset_chunks(ds, *, num_classes: int, chunk_size: int = 128):
    for begin in range(0, len(ds), chunk_size):
        end = min(begin + chunk_size, len(ds))
        count = end - begin
        examples = np.empty(
            (count, 3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE),
            dtype=np.float16,
        )
        labels = np.zeros((count, num_classes), dtype=np.float16)
        for offset, source_index in enumerate(range(begin, end)):
            example = ds[source_index]
            label = int(example["label"])
            if label < 0 or label >= num_classes:
                raise RuntimeError(f"source index {source_index}: label {label} is outside [0, {num_classes})")
            examples[offset] = _center_crop_resize_to_chw_fp16(
                example["image"],
                image_size=IMAGENET100_IMAGE_SIZE,
                resize_shorter_side=IMAGENET100_RESIZE_SHORTER_SIDE,
            )
            labels[offset, label] = np.float16(1.0)
        yield {"examples": examples, "labels": labels}


def _ensure_imagenet100_dataset():
    IMAGENET100_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest_if_valid(IMAGENET100_CACHE_DIR)
    if manifest is not None:
        return manifest

    load_dataset, _ = _import_imagenet_dependencies()
    hf_cache = IMAGENET100_CACHE_DIR / "hf_datasets"
    train = load_dataset(IMAGENET100_DATASET_ID, split="train", cache_dir=str(hf_cache))
    validate = load_dataset(IMAGENET100_DATASET_ID, split="validation", cache_dir=str(hf_cache))
    label_feature = train.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    num_classes = len(label_names) if label_names else int(max(train["label"])) + 1
    if num_classes != IMAGENET100_NUM_CLASSES:
        raise RuntimeError(f"expected {IMAGENET100_NUM_CLASSES} ImageNet-100 classes, got {num_classes}")

    dataset_path = _imagenet100_dataset_path(IMAGENET100_CACHE_DIR)
    split_manifest_path = _imagenet100_split_manifest_path(IMAGENET100_CACHE_DIR)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    split_manifest_path.unlink(missing_ok=True)
    for fold_index in range(5):
        _imagenet100_cv5_split_manifest_path(IMAGENET100_CACHE_DIR, fold_index).unlink(missing_ok=True)

    train_count = len(train)
    validate_count = len(validate)
    if train_count != IMAGENET100_TRAIN_EXAMPLES:
        raise RuntimeError(f"expected {IMAGENET100_TRAIN_EXAMPLES} ImageNet-100 train examples, got {train_count}")
    if validate_count != IMAGENET100_VALIDATE_EXAMPLES:
        raise RuntimeError(
            f"expected {IMAGENET100_VALIDATE_EXAMPLES} ImageNet-100 validation examples, got {validate_count}")
    dataset = write_dense_indexed_dataset(
        dataset_path=dataset_path,
        tensor_shapes={
            "examples": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
            "labels": [num_classes],
        },
        data_type=thor.DataType.fp16,
        chunks=(
            chunk
            for split in (train, validate)
            for chunk in _imagenet100_dataset_chunks(split, num_classes=num_classes)
        ),
        expected_num_examples=train_count + validate_count,
        num_shards=IMAGENET100_NUM_SHARDS,
    )
    validate_indices = np.arange(train_count, train_count + validate_count, dtype=np.int64)
    save_split_manifest(
        dataset=dataset,
        path=split_manifest_path,
        train_indices=np.arange(0, train_count, dtype=np.int64),
        validate_indices=validate_indices,
        test_indices=validate_indices,
    )

    manifest = _base_manifest(
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        num_classes=num_classes,
        train_examples=train_count,
        validate_examples=validate_count,
        label_names=label_names,
    )
    _manifest_path(IMAGENET100_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _imagenet100_data_from_manifest(
    manifest: dict,
    *,
    split_manifest_path: Path,
    batch_size: int,
    dataset_name: str,
):
    return open_training_data(
        dataset_path=Path(manifest["dataset_path"]),
        split_manifest_path=split_manifest_path,
        batch_size=batch_size,
        dataset_name=dataset_name,
        randomize_train=True,
        device_storage="off",
    )


def _imagenet100_data(*, batch_size: int):
    manifest = _ensure_imagenet100_dataset()
    data = _imagenet100_data_from_manifest(
        manifest,
        split_manifest_path=Path(manifest["split_manifest_path"]),
        batch_size=batch_size,
        dataset_name="clane9_imagenet100_preprocessed_fp16_chw",
    )
    return data, manifest


def _imagenet100_cv5_manifest_path(cache_root: Path) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_fp16_cv5_manifest.json"


def _imagenet100_cv5_split_manifest_path(cache_root: Path, fold_index: int) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_cv5_fold_{fold_index}_split_v2.json"


def _imagenet100_cv5_base_manifest(
    *,
    dataset_path: Path,
    source_examples: int,
    cv_examples: int,
    test_examples: int,
    num_classes: int,
    label_names: list[str],
    folds: list[dict],
) -> dict:
    return {
        "version": IMAGENET100_CV5_MANIFEST_VERSION,
        "source_version": IMAGENET100_MANIFEST_VERSION,
        "dataset_id": IMAGENET100_DATASET_ID,
        "split_source": "train",
        "image_size": IMAGENET100_IMAGE_SIZE,
        "resize_shorter_side": IMAGENET100_RESIZE_SHORTER_SIDE,
        "dtype": "fp16",
        "num_classes": num_classes,
        "num_folds": 5,
        "holdout_test_fraction": IMAGENET100_CV5_HOLDOUT_TEST_FRACTION,
        "max_examples_per_class": IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS,
        "source_examples": source_examples,
        "cv_examples": cv_examples,
        "test_examples": test_examples,
        "example_shape": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
        "label_shape": [num_classes],
        "label_names": list(label_names),
        "dataset_path": str(dataset_path),
        "folds": folds,
    }


def _read_imagenet100_cv5_manifest_if_valid(cache_root: Path):
    manifest_file = _imagenet100_cv5_manifest_path(cache_root)
    if IMAGENET100_REBUILD or IMAGENET100_CV5_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None

    expected = {
        "version": IMAGENET100_CV5_MANIFEST_VERSION,
        "source_version": IMAGENET100_MANIFEST_VERSION,
        "dataset_id": IMAGENET100_DATASET_ID,
        "split_source": "train",
        "image_size": IMAGENET100_IMAGE_SIZE,
        "resize_shorter_side": IMAGENET100_RESIZE_SHORTER_SIDE,
        "dtype": "fp16",
        "num_folds": 5,
        "holdout_test_fraction": IMAGENET100_CV5_HOLDOUT_TEST_FRACTION,
        "max_examples_per_class": IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS,
        "example_shape": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    dataset_path = manifest.get("dataset_path")
    if not dataset_path or not (Path(dataset_path) / "manifest.json").exists():
        return None
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        return None
    source_examples = manifest.get("source_examples")
    cv_examples = manifest.get("cv_examples")
    test_examples = manifest.get("test_examples")
    if not isinstance(source_examples, int) or source_examples <= 0:
        return None
    if not isinstance(cv_examples, int) or cv_examples <= 0:
        return None
    if not isinstance(test_examples, int) or test_examples <= 0:
        return None
    if cv_examples + test_examples != source_examples:
        return None
    if manifest.get("label_shape") != [manifest.get("num_classes")]:
        return None
    for fold in folds:
        split_manifest_path = fold.get("split_manifest_path") if isinstance(fold, dict) else None
        if not split_manifest_path or not Path(split_manifest_path).exists():
            return None
        if fold.get("test_examples") != test_examples:
            return None
    return manifest


def _stratified_fold_indices(labels: np.ndarray, *, num_classes: int, num_folds: int) -> list[np.ndarray]:
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(num_folds)]
    for label in range(num_classes):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.shape[0] < num_folds:
            raise RuntimeError(
                f"class {label} has only {label_indices.shape[0]} examples, cannot make {num_folds} stratified folds")
        for fold_index in range(num_folds):
            fold_parts[fold_index].append(label_indices[fold_index::num_folds])
    folds = []
    for fold_index in range(num_folds):
        fold_indices = np.concatenate(fold_parts[fold_index])
        fold_indices.sort()
        folds.append(fold_indices)
    return folds


def _stratified_holdout_indices(labels: np.ndarray, *, num_classes: int, fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise RuntimeError(f"holdout fraction must be between 0 and 1, got {fraction}")

    total_count = int(labels.shape[0])
    target_count = int(round(total_count * fraction))
    if target_count <= 0 or target_count >= total_count:
        raise RuntimeError(f"invalid holdout target count {target_count} for {total_count} examples")

    label_parts = []
    allocated = 0
    for label in range(num_classes):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.shape[0] == 0:
            raise RuntimeError(f"class {label} has no examples")
        exact_count = float(label_indices.shape[0]) * fraction
        base_count = int(math.floor(exact_count))
        label_parts.append(
            {
                "label": label,
                "indices": label_indices,
                "count": base_count,
                "remainder": exact_count - base_count,
            })
        allocated += base_count

    remaining = target_count - allocated
    if remaining > 0:
        for label_part in sorted(label_parts, key=lambda part: (-part["remainder"], part["label"]))[:remaining]:
            label_part["count"] += 1

    holdout_parts = []
    for label_part in label_parts:
        count = int(label_part["count"])
        if count <= 0:
            raise RuntimeError(
                f"class {label_part['label']} would contribute no holdout examples; use more data or a larger fraction")
        label_indices = label_part["indices"]
        stride = max(1, int(round(1.0 / fraction)))
        selected = label_indices[::stride]
        if selected.shape[0] < count:
            selected = label_indices
        holdout_parts.append(selected[:count])

    holdout_indices = np.concatenate(holdout_parts)
    holdout_indices.sort()
    if holdout_indices.shape[0] != target_count:
        raise RuntimeError(f"holdout split selected {holdout_indices.shape[0]} examples, expected {target_count}")
    return holdout_indices


def _imagenet100_cv5_source_indices(labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    if IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS <= 0:
        return np.arange(labels.shape[0], dtype=np.int64)

    parts = []
    for label in range(num_classes):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.shape[0] == 0:
            raise RuntimeError(f"class {label} has no examples")
        parts.append(label_indices[:IMAGENET100_CV5_MAX_EXAMPLES_PER_CLASS])
    source_indices = np.concatenate(parts).astype(np.int64, copy=False)
    source_indices.sort()
    return source_indices


def _ensure_imagenet100_cv5_manifests():
    IMAGENET100_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_imagenet100_cv5_manifest_if_valid(IMAGENET100_CACHE_DIR)
    if manifest is not None:
        return manifest

    base_manifest = _ensure_imagenet100_dataset()
    dataset_path = Path(base_manifest["dataset_path"])
    dataset = thor.data.FileDataset.open(dataset_path)
    load_dataset, _ = _import_imagenet_dependencies()
    hf_cache = IMAGENET100_CACHE_DIR / "hf_datasets"
    train = load_dataset(IMAGENET100_DATASET_ID, split="train", cache_dir=str(hf_cache))
    label_feature = train.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    labels = np.asarray(train["label"], dtype=np.int64)
    num_classes = len(label_names) if label_names else int(np.max(labels)) + 1
    if num_classes != IMAGENET100_NUM_CLASSES:
        raise RuntimeError(f"expected {IMAGENET100_NUM_CLASSES} ImageNet-100 classes, got {num_classes}")

    source_indices = _imagenet100_cv5_source_indices(labels, num_classes=num_classes)
    source_labels = labels[source_indices]
    holdout_relative_indices = _stratified_holdout_indices(
        source_labels,
        num_classes=num_classes,
        fraction=IMAGENET100_CV5_HOLDOUT_TEST_FRACTION,
    )
    selected_mask = np.ones(source_indices.shape[0], dtype=bool)
    selected_mask[holdout_relative_indices] = False
    cv_indices = source_indices[selected_mask]
    holdout_indices = source_indices[holdout_relative_indices]
    fold_indices = _stratified_fold_indices(labels[cv_indices], num_classes=num_classes, num_folds=5)

    folds = []
    for fold_index, validate_relative_indices in enumerate(fold_indices):
        fold_cv_train_mask = np.ones(cv_indices.shape[0], dtype=bool)
        fold_cv_train_mask[validate_relative_indices] = False
        train_indices = cv_indices[fold_cv_train_mask]
        validate_indices = cv_indices[validate_relative_indices]
        split_manifest_path = _imagenet100_cv5_split_manifest_path(IMAGENET100_CACHE_DIR, fold_index)
        save_split_manifest(
            dataset=dataset,
            path=split_manifest_path,
            train_indices=train_indices,
            validate_indices=validate_indices,
            test_indices=holdout_indices,
        )
        folds.append(
            {
                "fold_index": fold_index,
                "train_examples": int(train_indices.shape[0]),
                "validate_examples": int(validate_indices.shape[0]),
                "test_examples": int(holdout_indices.shape[0]),
                "split_manifest_path": str(split_manifest_path),
            })

    manifest = _imagenet100_cv5_base_manifest(
        dataset_path=dataset_path,
        source_examples=int(source_indices.shape[0]),
        cv_examples=int(cv_indices.shape[0]),
        test_examples=int(holdout_indices.shape[0]),
        num_classes=num_classes,
        label_names=label_names,
        folds=folds,
    )
    _imagenet100_cv5_manifest_path(IMAGENET100_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _object_detection_image_elems(image_size: int) -> int:
    return 3 * image_size * image_size


def _object_detection_example_elems(image_size: int) -> int:
    return _object_detection_image_elems(image_size)


def _object_detection_manifest_path(cache_root: Path) -> Path:
    return cache_root / f"voc2012_detection_{OBJECT_DETECTION_IMAGE_SIZE}_numpy_fp16_manifest.json"


def _object_detection_arrays_root(cache_root: Path) -> Path:
    return cache_root / f"voc2012_detection_arrays_{OBJECT_DETECTION_IMAGE_SIZE}_fp16"


def _object_detection_array_paths(arrays_root: Path) -> dict[str, str]:
    return {
        "train_examples": str(arrays_root / "train_examples.npy"),
        "train_labels": str(arrays_root / "train_boxes.npy"),
        "validate_examples": str(arrays_root / "validate_examples.npy"),
        "validate_labels": str(arrays_root / "validate_boxes.npy"),
    }


def _object_detection_base_manifest(*, array_paths: dict[str, str], train_examples: int, validate_examples: int):
    image_size = OBJECT_DETECTION_IMAGE_SIZE
    return {
        "version": OBJECT_DETECTION_MANIFEST_VERSION,
        "dataset_url": OBJECT_DETECTION_DATASET_URL,
        "image_size": image_size,
        "dtype": "fp16",
        "box_format": "xyxy_normalized",
        "max_train_examples": OBJECT_DETECTION_TRAIN_EXAMPLES,
        "max_validate_examples": OBJECT_DETECTION_VALIDATE_EXAMPLES,
        "num_classes": len(OBJECT_DETECTION_VOC_CLASSES),
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": validate_examples,
        "example_shape": [3, image_size, image_size],
        "image_shape": [3, image_size, image_size],
        "box_shape": [OBJECT_DETECTION_BOX_DIMS],
        "label_shape": [OBJECT_DETECTION_BOX_DIMS],
        "array_paths": {name: str(Path(path)) for name, path in sorted(array_paths.items())},
        "label_names": ["x1", "y1", "x2", "y2"],
        "class_names": list(OBJECT_DETECTION_VOC_CLASSES),
    }


def _read_object_detection_manifest_if_valid(cache_root: Path):
    manifest_file = _object_detection_manifest_path(cache_root)
    if OBJECT_DETECTION_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None

    expected = {
        "version": OBJECT_DETECTION_MANIFEST_VERSION,
        "dataset_url": OBJECT_DETECTION_DATASET_URL,
        "image_size": OBJECT_DETECTION_IMAGE_SIZE,
        "dtype": "fp16",
        "box_format": "xyxy_normalized",
        "max_train_examples": OBJECT_DETECTION_TRAIN_EXAMPLES,
        "max_validate_examples": OBJECT_DETECTION_VALIDATE_EXAMPLES,
        "label_shape": [OBJECT_DETECTION_BOX_DIMS],
        "example_shape": [3, OBJECT_DETECTION_IMAGE_SIZE, OBJECT_DETECTION_IMAGE_SIZE],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    array_paths = manifest.get("array_paths") or {}
    if sorted(array_paths) != ["train_examples", "train_labels", "validate_examples", "validate_labels"]:
        return None
    for path in array_paths.values():
        if not Path(path).exists():
            return None
    return manifest


def _import_object_detection_dependencies():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The heavyweight object-detection integration tests require the optional "
            "Python package 'Pillow'. Install it in the test venv before setting "
            "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1 or THOR_ALL_INTEGRATION_TESTS=1."
        ) from exc
    return Image


def _safe_extract_tar(archive: tarfile.TarFile, dest_dir: Path):
    dest_resolved = dest_dir.resolve()
    for member in archive.getmembers():
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise RuntimeError(f"refusing to extract path outside cache directory: {member.name}")
    archive.extractall(dest_resolved)


def _ensure_voc2012_trainval_downloaded(cache_root: Path) -> Path:
    voc_root = cache_root / "VOCdevkit" / "VOC2012"
    if voc_root.exists():
        return voc_root

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / "VOCtrainval_11-May-2012.tar"
    if not archive_path.exists() or OBJECT_DETECTION_REBUILD:
        urllib.request.urlretrieve(OBJECT_DETECTION_DATASET_URL, archive_path)

    with tarfile.open(archive_path, "r") as archive:
        _safe_extract_tar(archive, cache_root)

    if not voc_root.exists():
        raise RuntimeError(f"VOC2012 archive extraction did not create expected directory {voc_root}")
    return voc_root


def _voc_split_ids(voc_root: Path, split_name: str) -> list[str]:
    split_file = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"
    ids = [line.strip().split()[0] for line in split_file.read_text().splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"VOC2012 split {split_name} is empty: {split_file}")
    return ids


def _read_voc_primary_object(annotation_path: Path):
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        return None
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 1 or height <= 1:
        return None

    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in OBJECT_DETECTION_VOC_CLASSES:
            continue
        box_node = obj.find("bndbox")
        if box_node is None:
            continue
        try:
            xmin = float(box_node.findtext("xmin", "0")) - 1.0
            ymin = float(box_node.findtext("ymin", "0")) - 1.0
            xmax = float(box_node.findtext("xmax", "0"))
            ymax = float(box_node.findtext("ymax", "0"))
        except ValueError:
            continue

        x1 = min(max(xmin, 0.0), float(width - 1)) / float(width)
        y1 = min(max(ymin, 0.0), float(height - 1)) / float(height)
        x2 = min(max(xmax, 1.0), float(width)) / float(width)
        y2 = min(max(ymax, 1.0), float(height)) / float(height)
        min_extent = 1.0 / float(max(width, height))
        x2 = min(1.0, max(x2, x1 + min_extent))
        y2 = min(1.0, max(y2, y1 + min_extent))
        if x2 <= x1 or y2 <= y1:
            continue
        return class_name, np.asarray([x1, y1, x2, y2], dtype=np.float16)
    return None


def _resize_to_chw_fp16(image, *, image_size: int) -> np.ndarray:
    from PIL import Image  # type: ignore

    image = image.convert("RGB")
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((image_size, image_size), resampling)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr, dtype=np.float16)


def _write_voc_detection_split(voc_root: Path, *, split_name: str, split_ids: list[str], raw_root: Path, image_size: int) -> int:
    Image = _import_object_detection_dependencies()
    for class_name in OBJECT_DETECTION_VOC_CLASSES:
        (raw_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    num_written = 0
    for image_id in split_ids:
        primary = _read_voc_primary_object(voc_root / "Annotations" / f"{image_id}.xml")
        if primary is None:
            continue
        class_name, box = primary
        with Image.open(voc_root / "JPEGImages" / f"{image_id}.jpg") as image:
            chw = _resize_to_chw_fp16(image, image_size=image_size)
        packed = np.concatenate([chw.reshape(-1), box.reshape(-1)]).astype(np.float16, copy=False)
        filename = raw_root / split_name / class_name / f"{split_name}_{image_id}.bin"
        filename.write_bytes(np.ascontiguousarray(packed).tobytes(order="C"))
        num_written += 1

    if num_written == 0:
        raise RuntimeError(f"VOC2012 split {split_name} produced no usable detection examples")
    return num_written


def _mirror_voc_validate_as_test(raw_root: Path):
    for class_name in OBJECT_DETECTION_VOC_CLASSES:
        validate_dir = raw_root / "validate" / class_name
        test_dir = raw_root / "test" / class_name
        test_dir.mkdir(parents=True, exist_ok=True)
        for validate_file in validate_dir.iterdir():
            if not validate_file.is_file():
                continue
            test_file = test_dir / validate_file.name.replace("validate_", "test_", 1)
            try:
                os.link(validate_file, test_file)
            except OSError:
                shutil.copy2(validate_file, test_file)


def _load_voc_detection_numpy_split(voc_root: Path, *, split_name: str, split_ids: list[str], max_examples: int, image_size: int):
    Image = _import_object_detection_dependencies()
    examples: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    class_counts: dict[str, int] = {}

    for image_id in split_ids:
        primary = _read_voc_primary_object(voc_root / "Annotations" / f"{image_id}.xml")
        if primary is None:
            continue
        class_name, box = primary
        with Image.open(voc_root / "JPEGImages" / f"{image_id}.jpg") as image:
            examples.append(_resize_to_chw_fp16(image, image_size=image_size))
        boxes.append(np.ascontiguousarray(box, dtype=np.float16))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if len(examples) >= max_examples:
            break

    if not examples:
        raise RuntimeError(f"VOC2012 split {split_name} produced no usable detection examples")
    return (
        np.ascontiguousarray(np.stack(examples, axis=0), dtype=np.float16),
        np.ascontiguousarray(np.stack(boxes, axis=0), dtype=np.float16),
        class_counts,
    )


def _ensure_voc_detection_arrays():
    OBJECT_DETECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_object_detection_manifest_if_valid(OBJECT_DETECTION_CACHE_DIR)
    if manifest is not None:
        return manifest

    voc_root = _ensure_voc2012_trainval_downloaded(OBJECT_DETECTION_CACHE_DIR)
    arrays_root = _object_detection_arrays_root(OBJECT_DETECTION_CACHE_DIR)
    if arrays_root.exists():
        shutil.rmtree(arrays_root)
    arrays_root.mkdir(parents=True, exist_ok=True)

    train_examples, train_boxes, train_class_counts = _load_voc_detection_numpy_split(
        voc_root,
        split_name="train",
        split_ids=_voc_split_ids(voc_root, "train"),
        max_examples=OBJECT_DETECTION_TRAIN_EXAMPLES,
        image_size=OBJECT_DETECTION_IMAGE_SIZE,
    )
    validate_examples, validate_boxes, validate_class_counts = _load_voc_detection_numpy_split(
        voc_root,
        split_name="validate",
        split_ids=_voc_split_ids(voc_root, "val"),
        max_examples=OBJECT_DETECTION_VALIDATE_EXAMPLES,
        image_size=OBJECT_DETECTION_IMAGE_SIZE,
    )

    array_paths = _object_detection_array_paths(arrays_root)
    np.save(array_paths["train_examples"], train_examples)
    np.save(array_paths["train_labels"], train_boxes)
    np.save(array_paths["validate_examples"], validate_examples)
    np.save(array_paths["validate_labels"], validate_boxes)

    manifest = _object_detection_base_manifest(
        array_paths=array_paths,
        train_examples=int(train_examples.shape[0]),
        validate_examples=int(validate_examples.shape[0]),
    )
    manifest["train_class_counts"] = train_class_counts
    manifest["validate_class_counts"] = validate_class_counts
    _object_detection_manifest_path(OBJECT_DETECTION_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _voc_detection_loader(*, batch_size: int):
    manifest = _ensure_voc_detection_arrays()
    array_paths = manifest["array_paths"]
    loader = make_numpy_pair_training_data(
        np.ascontiguousarray(np.load(array_paths["train_examples"]), dtype=np.float16),
        np.ascontiguousarray(np.load(array_paths["train_labels"]), dtype=np.float16),
        np.ascontiguousarray(np.load(array_paths["validate_examples"]), dtype=np.float16),
        np.ascontiguousarray(np.load(array_paths["validate_labels"]), dtype=np.float16),
        batch_size=batch_size,
        dataset_name="pascal_voc2012_detection_numpy_fp16_chw_xyxy",
    )
    return loader, manifest


def _serializable_relu(network: thor.Network, x: thor.Tensor, *, dtype=thor.DataType.fp16) -> thor.Tensor:
    return thor.activations.Relu().add_to_network(network, x)


def _serializable_residual_add_relu(
    network: thor.Network,
    main: thor.Tensor,
    shortcut: thor.Tensor,
    *,
    dtype=thor.DataType.fp16,
) -> thor.Tensor:
    main_expr = thor.activations.Activation.epilogue_input(output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    shortcut_expr = thor.activations.Activation.epilogue_aux_input(
        "shortcut", output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    y = thor.activations.Relu().to_expression(main_expr + shortcut_expr).with_dtypes(
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    return thor.activations.Relu().add_to_network(
        network,
        main,
        epilogue=y,
        epilogue_inputs={"shortcut": shortcut},
    )


def _conv_bn_relu(
    network: thor.Network,
    x: thor.Tensor,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dtype=thor.DataType.fp16,
) -> thor.Tensor:
    conv = thor.layers.Convolution2d(
        network,
        x,
        out_channels,
        kernel_size,
        kernel_size,
        vertical_stride=stride,
        horizontal_stride=stride,
        vertical_padding=padding,
        horizontal_padding=padding,
        has_bias=False,
        activation=None,
    )
    bn = thor.layers.BatchNormalization(network, conv.get_feature_output())
    return _serializable_relu(network, bn.get_feature_output(), dtype=dtype)


def _resnet18_basic_block(
    network: thor.Network,
    x: thor.Tensor,
    out_channels: int,
    *,
    stride: int,
    dtype=thor.DataType.fp16,
) -> thor.Tensor:
    conv1 = thor.layers.Convolution2d(
        network,
        x,
        out_channels,
        3,
        3,
        vertical_stride=stride,
        horizontal_stride=stride,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=False,
        activation=None,
    )
    bn1 = thor.layers.BatchNormalization(network, conv1.get_feature_output())
    relu1 = _serializable_relu(network, bn1.get_feature_output(), dtype=dtype)

    conv2 = thor.layers.Convolution2d(
        network,
        relu1,
        out_channels,
        3,
        3,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=False,
        activation=None,
    )
    bn2 = thor.layers.BatchNormalization(network, conv2.get_feature_output())
    main = bn2.get_feature_output()

    in_channels = x.get_dimensions()[0]
    if stride != 1 or in_channels != out_channels:
        shortcut_conv = thor.layers.Convolution2d(
            network,
            x,
            out_channels,
            1,
            1,
            vertical_stride=stride,
            horizontal_stride=stride,
            has_bias=False,
            activation=None,
        )
        shortcut_bn = thor.layers.BatchNormalization(network, shortcut_conv.get_feature_output())
        shortcut = shortcut_bn.get_feature_output()
    else:
        shortcut = x

    return _serializable_residual_add_relu(network, main, shortcut, dtype=dtype)


def _build_alexnet_imagenet100(name: str, *, num_classes: int, dtype=thor.DataType.fp16):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)

    x = examples.get_feature_output()
    conv1 = thor.layers.Convolution2d(network, x, 64, 11, 11, 4, 4, 2, 2, True, activation=thor.activations.Relu())
    x = conv1.get_feature_output()
    pool1 = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2)
    x = pool1.get_feature_output()

    conv2 = thor.layers.Convolution2d(network, x, 192, 5, 5, 1, 1, 2, 2, True, activation=thor.activations.Relu())
    x = conv2.get_feature_output()
    pool2 = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2)
    x = pool2.get_feature_output()

    conv3 = thor.layers.Convolution2d(network, x, 384, 3, 3, 1, 1, 1, 1, True, activation=thor.activations.Relu())
    x = conv3.get_feature_output()
    conv4 = thor.layers.Convolution2d(network, x, 256, 3, 3, 1, 1, 1, 1, True, activation=thor.activations.Relu())
    x = conv4.get_feature_output()
    conv5 = thor.layers.Convolution2d(network, x, 256, 3, 3, 1, 1, 1, 1, True, activation=thor.activations.Relu())
    x = conv5.get_feature_output()
    pool5 = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2)
    x = pool5.get_feature_output()

    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    fc6 = thor.layers.FullyConnected(network, x, 4096, True, activation=thor.activations.Relu())
    x = fc6.get_feature_output()
    drop6 = thor.layers.DropOut(network, x, 0.5)
    x = drop6.get_feature_output()
    fc7 = thor.layers.FullyConnected(network, x, 4096, True, activation=thor.activations.Relu())
    x = fc7.get_feature_output()
    drop7 = thor.layers.DropOut(network, x, 0.5)
    x = drop7.get_feature_output()
    logits = thor.layers.FullyConnected(network, x, num_classes, True, activation=None)

    loss = thor.losses.CategoricalCrossEntropy(
        network, logits.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def _build_resnet18_imagenet100(name: str, *, num_classes: int, dtype=thor.DataType.fp16):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)

    x = _conv_bn_relu(network, examples.get_feature_output(), 64, 7, stride=2, padding=3, dtype=dtype)
    pool = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 3, 3, 2, 2, 1, 1)
    x = pool.get_feature_output()

    for out_channels, strides in ((64, (1, 1)), (128, (2, 1)), (256, (2, 1)), (512, (2, 1))):
        for stride in strides:
            x = _resnet18_basic_block(network, x, out_channels, stride=stride, dtype=dtype)

    avgpool = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.average, 7, 7, 1, 1)
    x = avgpool.get_feature_output()
    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    logits = thor.layers.FullyConnected(network, x, num_classes, True, activation=None)

    loss = thor.losses.CategoricalCrossEntropy(
        network, logits.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def _global_average_pool_2d(network: thor.Network, x: thor.Tensor) -> thor.Tensor:
    _, height, width = x.get_dimensions()
    pool = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.average, height, width, 1, 1)
    return pool.get_feature_output()


def _build_tiny_voc2012_box_detector(
    name: str,
    *,
    num_classes: int,
    image_size: int = OBJECT_DETECTION_IMAGE_SIZE,
    dtype=thor.DataType.fp16,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [3, image_size, image_size], dtype)
    target_box = thor.layers.NetworkInput(network, "labels", [OBJECT_DETECTION_BOX_DIMS], dtype)
    image = examples.get_feature_output()

    x = _conv_bn_relu(network, image, 32, 3, padding=1, dtype=dtype)
    pool1 = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 2, 2, 2, 2)
    x = pool1.get_feature_output()
    x = _conv_bn_relu(network, x, 64, 3, padding=1, dtype=dtype)
    pool2 = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.max, 2, 2, 2, 2)
    x = pool2.get_feature_output()
    x = _conv_bn_relu(network, x, 128, 3, padding=1, dtype=dtype)
    x = _global_average_pool_2d(network, x)
    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    hidden = thor.layers.FullyConnected(network, x, 256, True, activation=thor.activations.Relu())
    x = hidden.get_feature_output()
    drop = thor.layers.DropOut(network, x, 0.10)
    x = drop.get_feature_output()

    raw_box = thor.layers.FullyConnected(network, x, OBJECT_DETECTION_BOX_DIMS, True, activation=None)
    pred_box = thor.activations.Sigmoid().add_to_network(network, raw_box.get_feature_output())

    box_loss = thor.losses.detection.CIoULoss(
        network,
        pred_box,
        target_box.get_feature_output(),
        "xyxy",
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", box_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "pred_boxes", pred_box, thor.DataType.fp32)
    return network


def _run_full_imagenet100_model_training(model_builder, *, model_name: str, capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        data, manifest = _imagenet100_data(batch_size=IMAGENET100_BATCH_SIZE)
        assert manifest["train_examples"] == len(data.splits.train)
        assert manifest["validate_examples"] == len(data.splits.validate)
        assert len(data.splits.train) >= IMAGENET100_BATCH_SIZE, "ImageNet-100 train split unexpectedly has no full batch"
        assert len(data.splits.validate) >= IMAGENET100_BATCH_SIZE, "ImageNet-100 validation split unexpectedly has no full batch"
        assert manifest["num_classes"] == manifest["label_shape"][0]

        network = model_builder(f"python_integration_{model_name}_imagenet100_full", num_classes=manifest["num_classes"])
        optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.9)
        trainer = thor.training.Trainer(
            network,
            data=data,
            optimizer=optimizer,
            debug_synchronous=False,
            stats_interval_s=IMAGENET100_STATS_INTERVAL_S,
            max_in_flight_batches=IMAGENET100_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=IMAGENET100_EPOCHS)
        _assert_finite_positive_losses(stats, model_name=model_name)


def _run_full_voc2012_detection_model_training(model_builder, *, model_name: str, capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _voc_detection_loader(batch_size=OBJECT_DETECTION_BATCH_SIZE)
        session = loader.open_session()
        assert manifest["train_examples"] == session.get_num_train_examples()
        assert manifest["validate_examples"] == session.get_num_validate_examples()
        assert session.get_num_train_batches() > 0, "VOC2012 detection train split unexpectedly has zero batches"
        assert session.get_num_validate_batches() > 0, "VOC2012 detection validation split unexpectedly has zero batches"
        assert manifest["label_shape"] == [OBJECT_DETECTION_BOX_DIMS]
        assert manifest["box_shape"] == [OBJECT_DETECTION_BOX_DIMS]

        network = model_builder(
            f"python_integration_{model_name}_voc2012_detection_full",
            num_classes=manifest["num_classes"],
            image_size=manifest["image_size"],
        )
        optimizer = thor.optimizers.AdamW(alpha=0.001, weight_decay=0.01)
        trainer = thor.training.Trainer(
            network,
            data=loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats_interval_s=OBJECT_DETECTION_STATS_INTERVAL_S,
            max_in_flight_batches=OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=OBJECT_DETECTION_EPOCHS)
        _assert_finite_positive_losses(stats, model_name=model_name)


def _imagenet100_cv5_optimizer():
    # Classic SGD+momentum is the most robust default for these fp16 convolutional
    # ImageNet-style models. The small per-epoch decay keeps longer CV runs from
    # over-driving the later epochs while leaving 1-epoch smoke runs effectively unchanged.
    return thor.optimizers.Sgd(
        initial_learning_rate=IMAGENET100_CV5_LEARNING_RATE,
        decay=IMAGENET100_CV5_DECAY,
        momentum=IMAGENET100_CV5_MOMENTUM,
        nesterov_momentum=True,
    )


def _run_imagenet100_cv5_training_runs(model_builder, *, model_name: str, capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    ensemble_group = f"imagenet100_cv5_{model_name}"
    with capfd.disabled():
        cv_manifest = _ensure_imagenet100_cv5_manifests()
        assert cv_manifest["num_folds"] == 5
        assert len(cv_manifest["folds"]) == 5
        assert cv_manifest["example_shape"] == [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE]
        assert cv_manifest["label_shape"] == [cv_manifest["num_classes"]]
        assert cv_manifest["test_examples"] == int(
            round(cv_manifest["source_examples"] * IMAGENET100_CV5_HOLDOUT_TEST_FRACTION))
        assert cv_manifest["cv_examples"] + cv_manifest["test_examples"] == cv_manifest["source_examples"]

        def make_fold_trainer(*, fold: dict, run_name: str, save_model_dir: Path):
            fold_index = int(fold["fold_index"])
            data = _imagenet100_data_from_manifest(
                cv_manifest,
                split_manifest_path=Path(fold["split_manifest_path"]),
                batch_size=IMAGENET100_CV5_BATCH_SIZE,
                dataset_name=f"clane9_imagenet100_fp16_cv5_{model_name}_{run_name}",
            )
            assert fold["train_examples"] == len(data.splits.train)
            assert fold["validate_examples"] == len(data.splits.validate)
            assert fold["test_examples"] == len(data.splits.test) == cv_manifest["test_examples"]
            assert len(data.splits.train) >= IMAGENET100_CV5_BATCH_SIZE
            assert len(data.splits.validate) >= IMAGENET100_CV5_BATCH_SIZE

            network = model_builder(
                f"python_integration_imagenet100_cv5_{model_name}_fold_{fold_index}",
                num_classes=cv_manifest["num_classes"],
            )
            return thor.training.Trainer(
                network,
                data=data,
                optimizer=_imagenet100_cv5_optimizer(),
                debug_synchronous=False,
                stats_interval_s=IMAGENET100_CV5_STATS_INTERVAL_S,
                max_in_flight_batches=IMAGENET100_CV5_MAX_IN_FLIGHT_BATCHES,
                scalar_tensors_to_report=["loss"],
                stats_color=IMAGENET100_CV5_STATS_COLOR,
                save_model_dir=str(save_model_dir),
                save_model_overwrite=True,
            )

        artifact_root = IMAGENET100_CV5_MODEL_ARTIFACTS_DIR / model_name
        run_specs = []
        for fold in cv_manifest["folds"]:
            fold_index = int(fold["fold_index"])
            trainer = make_fold_trainer(
                fold=fold,
                run_name=f"fold_{fold_index}",
                save_model_dir=artifact_root / f"fold_{fold_index}",
            )
            run_specs.append((f"fold_{fold_index}", trainer, ensemble_group))

        runs = thor.training.TrainingRuns(
            run_specs,
            max_summary_logs_per_second=IMAGENET100_CV5_SUMMARY_LOGS_PER_SECOND,
            max_parallel_runs=IMAGENET100_CV5_MAX_PARALLEL_RUNS,
        )
        test_fold = cv_manifest["folds"][0]
        test_data = _imagenet100_data_from_manifest(
            cv_manifest,
            split_manifest_path=Path(test_fold["split_manifest_path"]),
            batch_size=IMAGENET100_CV5_BATCH_SIZE,
            dataset_name=f"clane9_imagenet100_fp16_cv5_{model_name}_holdout_test",
        )
        results, captured_text = _fit_training_runs_and_capture_text(
            runs,
            epochs=IMAGENET100_CV5_EPOCHS,
            test_loader=test_data.open_session(max_in_flight_batches=IMAGENET100_CV5_MAX_IN_FLIGHT_BATCHES),
        )

    plain_text = _ANSI_RE.sub("", captured_text)
    statuses = _captured_run_statuses(captured_text)

    assert len(results) == 5
    assert results.all_completed()
    assert "INFO runs summary:" in plain_text
    assert "\nINFO runs final: ==================== final results" in plain_text
    assert "INFO runs final: total=5" in plain_text
    assert f"INFO runs ensemble[{ensemble_group}]:" in plain_text
    assert "aggregation=ensemble_eval" in plain_text
    assert "ensemble_train_loss=" in plain_text
    assert "ensemble_test_loss=" in plain_text
    assert "train_loss=" in plain_text
    assert "validate_loss=" in plain_text
    assert "test_loss=" in plain_text
    assert "test_accuracy=" not in plain_text
    assert results.status_counts["completed"] == 5
    assert results.has_ensembles
    assert len(results.ensembles) == 1

    ensemble = results.ensemble(ensemble_group)
    assert ensemble.all_completed()
    assert ensemble.total_weight == pytest.approx(5.0)
    assert len(ensemble.members) == 5
    assert ensemble.ensemble_train_loss is not None
    assert ensemble.ensemble_test_loss is not None
    assert math.isfinite(ensemble.ensemble_train_loss)
    assert math.isfinite(ensemble.ensemble_test_loss)
    assert ensemble.ensemble_train_loss > 0.0
    assert ensemble.ensemble_test_loss > 0.0

    validation_losses = []
    test_losses = []
    for fold_index in range(5):
        run_name = f"fold_{fold_index}"
        assert statuses[run_name] == "completed"
        assert re.search(
            rf"INFO runs\[{re.escape(run_name)}\|{re.escape(ensemble_group)}\]:.*"
            rf"train_loss=.*validate_loss=.*test_loss=",
            plain_text,
        ), f"final report did not include per-fold test_loss for {run_name}:\n{plain_text}"
        assert not re.search(
            rf"INFO runs\[{re.escape(run_name)}\|{re.escape(ensemble_group)}\]:.*test_accuracy=",
            plain_text,
        ), f"final report should not synthesize per-fold test_accuracy for {run_name}:\n{plain_text}"
        result = results[run_name]
        assert result.status == "completed"
        assert result.ensemble_group == ensemble_group
        assert result.ensemble_weight == pytest.approx(1.0)
        assert result.final_training_loss is not None
        assert result.final_validation_loss is not None
        assert result.final_test_loss is not None
        assert result.final_loss("train") == result.final_training_loss
        assert result.final_loss("validate") == result.final_validation_loss
        assert result.final_loss("test") == result.final_test_loss
        assert math.isfinite(result.final_training_loss)
        assert math.isfinite(result.final_validation_loss)
        assert math.isfinite(result.final_test_loss)
        assert result.final_training_loss > 0.0
        assert result.final_validation_loss > 0.0
        assert result.final_test_loss > 0.0
        validation_losses.append(result.final_validation_loss)
        test_losses.append(result.final_test_loss)

    assert math.isfinite(float(np.mean(validation_losses)))
    assert math.isfinite(float(np.mean(test_losses)))


@pytest.mark.skipif(
    not RUN_IMAGENET100_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION",
        description="VOC2012 object-detection training integration",
    ),
)
@pytest.mark.parametrize(
    ("model_name", "model_builder"),
    [
        ("tiny_box_detector", _build_tiny_voc2012_box_detector),
    ],
)
def test_queued_trainer_trains_voc2012_object_detection_networks_end_to_end(model_name, model_builder, capfd):
    _run_full_voc2012_detection_model_training(model_builder, model_name=model_name, capfd=capfd)


@pytest.mark.skipif(
    not RUN_IMAGENET100_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION",
        description="full AlexNet ImageNet-100 training integration",
    ),
)
def test_queued_trainer_trains_full_alexnet_on_full_clane9_imagenet100(capfd):
    _run_full_imagenet100_model_training(_build_alexnet_imagenet100, model_name="alexnet", capfd=capfd)


@pytest.mark.skipif(
    not RUN_IMAGENET100_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION",
        description="full ResNet-18 ImageNet-100 training integration",
    ),
)
def test_queued_trainer_trains_full_resnet18_on_full_clane9_imagenet100(capfd):
    _run_full_imagenet100_model_training(_build_resnet18_imagenet100, model_name="resnet18", capfd=capfd)


@pytest.mark.imagenet100_cv5_integration
@pytest.mark.skipif(
    not RUN_IMAGENET100_CV5_ALEXNET_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_CV5_ALEXNET_INTEGRATION",
        description="AlexNet ImageNet-100 CV5 TrainingRuns",
    ),
)
def test_training_runs_imagenet100_alexnet_five_fold_cross_validation(capfd):
    _run_imagenet100_cv5_training_runs(_build_alexnet_imagenet100, model_name="alexnet", capfd=capfd)


@pytest.mark.imagenet100_cv5_integration
@pytest.mark.skipif(
    not RUN_IMAGENET100_CV5_RESNET18_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_CV5_RESNET18_INTEGRATION",
        description="ResNet-18 ImageNet-100 CV5 TrainingRuns",
    ),
)
def test_training_runs_imagenet100_resnet18_five_fold_cross_validation(capfd):
    _run_imagenet100_cv5_training_runs(_build_resnet18_imagenet100, model_name="resnet18", capfd=capfd)
