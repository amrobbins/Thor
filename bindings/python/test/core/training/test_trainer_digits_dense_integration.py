import ctypes
import gzip
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import thor

RUN_DIGITS_DENSE_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_DIGITS_DENSE_INTEGRATION") == "1"
DIGITS_DENSE_CACHE_DIR = Path(os.environ.get("THOR_DIGITS_DENSE_CACHE_DIR", "/tmp/thor_digits_dense_training"))
DIGITS_DENSE_URL_BASE = os.environ.get(
    "THOR_DIGITS_DENSE_URL_BASE", "https://storage.googleapis.com/cvdf-datasets/mnist")
DIGITS_DENSE_BATCH_SIZE = int(os.environ.get("THOR_DIGITS_DENSE_BATCH_SIZE", "1024"))
DIGITS_DENSE_EPOCHS = int(os.environ.get("THOR_DIGITS_DENSE_EPOCHS", "1"))
DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES = int(os.environ.get("THOR_DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES", "8"))
DIGITS_DENSE_STATS_INTERVAL_S = float(os.environ.get("THOR_DIGITS_DENSE_STATS_INTERVAL_S", "0.0"))
DIGITS_DENSE_REBUILD = os.environ.get("THOR_DIGITS_DENSE_REBUILD") == "1"
DIGITS_DENSE_NUM_SHARDS = int(os.environ.get("THOR_DIGITS_DENSE_NUM_SHARDS", "1"))
DIGITS_DENSE_WIDTH = int(os.environ.get("THOR_DIGITS_DENSE_WIDTH", "8192"))
DIGITS_DENSE_HIDDEN_LAYERS = int(os.environ.get("THOR_DIGITS_DENSE_HIDDEN_LAYERS", "8"))
DIGITS_DENSE_MANIFEST_VERSION = 1
DIGITS_IMAGE_HEIGHT = 28
DIGITS_IMAGE_WIDTH = 28
DIGITS_INPUT_FEATURES = DIGITS_IMAGE_HEIGHT * DIGITS_IMAGE_WIDTH
DIGITS_NUM_CLASSES = 10
DIGITS_DOWNLOADS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "validate_images": "t10k-images-idx3-ubyte.gz",
    "validate_labels": "t10k-labels-idx1-ubyte.gz",
}

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.digits_dense_integration,
    pytest.mark.skipif(
        not RUN_DIGITS_DENSE_INTEGRATION,
        reason=(
            "set THOR_RUN_TRAINING_DIGITS_DENSE_INTEGRATION=1 to run the heavyweight "
            "fp16 dense DIGITS/MNIST training throughput test"),
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+).*?"
    r"flops/s=\s*(?P<flops>[-+0-9.eE]+[KMGTPE]?)")


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)


class _NativeOutputTee:
    """Mirror native stdout/stderr immediately while keeping text for assertions."""

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
                prefix=f"thor_digits_dense_fit_fd{fd}_", suffix=".log", delete=False)
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
                "flops_per_s": match.group("flops"),
            })
    return stats


def _flops_value(raw_value: str) -> float:
    suffix_scale = {
        "": 1.0,
        "K": 1.0e3,
        "M": 1.0e6,
        "G": 1.0e9,
        "T": 1.0e12,
        "P": 1.0e15,
        "E": 1.0e18,
    }
    suffix = raw_value[-1]
    if suffix.isalpha():
        return float(raw_value[:-1]) * suffix_scale[suffix]
    return float(raw_value)


def _fit_and_capture_stats(trainer, *, epochs: int):
    with _NativeOutputTee() as tee:
        trainer.fit(epochs=epochs)
    captured_text = tee.text()
    stats = _captured_trainer_stats(captured_text)

    assert stats, "trainer emitted no parseable stats; lower THOR_DIGITS_DENSE_STATS_INTERVAL_S if this happens"
    return stats


def _stats_phase_counts(stats):
    counts = {}
    for entry in stats:
        phase = entry["phase"]
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _assert_finite_positive_losses_and_flops(stats, *, model_name: str):
    losses = [entry["loss"] for entry in stats]
    phase_counts = _stats_phase_counts(stats)
    assert losses, f"{model_name}: no losses were reported; phase_counts={phase_counts}"
    for loss in losses:
        assert math.isfinite(loss), f"{model_name}: non-finite loss reported: {loss}; phase_counts={phase_counts}"
        assert loss > 0.0, f"{model_name}: non-positive loss reported: {loss}; phase_counts={phase_counts}"

    flops_values = [_flops_value(entry["flops_per_s"]) for entry in stats]
    assert flops_values, f"{model_name}: no FLOP/s values were reported; phase_counts={phase_counts}"
    assert max(flops_values) > 0.0, f"{model_name}: all reported FLOP/s values were zero; stats={stats}"
    assert any(
        entry["phase"] == "train"
        for entry in stats), f"{model_name}: no train stats reported; phase_counts={phase_counts}"
    assert any(entry["phase"] == "validate" for entry in stats), (
        f"{model_name}: no validate stats reported; phase_counts={phase_counts}. "
        "Re-run with THOR_DIGITS_DENSE_STATS_INTERVAL_S=0.0 if the validate phase finishes before the reporter interval."
    )


def _digits_manifest_path(cache_root: Path) -> Path:
    return cache_root / "mnist_digits_dense_fp16_manifest.json"


def _digits_shard_root(cache_root: Path) -> Path:
    return cache_root / "shards_fp16_flat"


def _download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)


def _ensure_mnist_downloads(cache_root: Path) -> dict[str, Path]:
    downloads_root = cache_root / "downloads"
    paths = {}
    for key, filename in DIGITS_DOWNLOADS.items():
        path = downloads_root / filename
        _download_if_missing(f"{DIGITS_DENSE_URL_BASE.rstrip('/')}/{filename}", path)
        paths[key] = path
    return paths


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise RuntimeError(f"{path} is not an IDX image file: magic={magic}")
        if rows != DIGITS_IMAGE_HEIGHT or cols != DIGITS_IMAGE_WIDTH:
            raise RuntimeError(f"unexpected DIGITS image shape in {path}: rows={rows}, cols={cols}")
        data = f.read()
    expected = count * rows * cols
    if len(data) != expected:
        raise RuntimeError(f"truncated IDX image file {path}: expected {expected} bytes, found {len(data)}")
    return np.frombuffer(data, dtype=np.uint8).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise RuntimeError(f"{path} is not an IDX label file: magic={magic}")
        data = f.read()
    if len(data) != count:
        raise RuntimeError(f"truncated IDX label file {path}: expected {count} bytes, found {len(data)}")
    labels = np.frombuffer(data, dtype=np.uint8)
    if np.any(labels >= DIGITS_NUM_CLASSES):
        raise RuntimeError(f"IDX labels out of range in {path}")
    return labels


def _class_dir(label: int) -> str:
    return f"class_{label}"


def _write_digits_split(images: np.ndarray, labels: np.ndarray, *, split_name: str, raw_root: Path) -> int:
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError(f"{split_name}: images/labels count mismatch: {images.shape[0]} != {labels.shape[0]}")
    for label in range(DIGITS_NUM_CLASSES):
        (raw_root / split_name / _class_dir(label)).mkdir(parents=True, exist_ok=True)

    for index, (image, label) in enumerate(zip(images, labels)):
        # Keep the examples flat so the benchmark's network body is strictly FullyConnected layers plus activations.
        flat = image.astype(np.float32).reshape(DIGITS_INPUT_FEATURES) / 255.0
        packed = np.ascontiguousarray(flat, dtype=np.float16)
        filename = raw_root / split_name / _class_dir(int(label)) / f"{split_name}_{index:05d}.bin"
        filename.write_bytes(packed.tobytes(order="C"))
    return int(images.shape[0])


def _mirror_validate_as_test(raw_root: Path):
    for label in range(DIGITS_NUM_CLASSES):
        validate_dir = raw_root / "validate" / _class_dir(label)
        test_dir = raw_root / "test" / _class_dir(label)
        test_dir.mkdir(parents=True, exist_ok=True)
        for validate_file in validate_dir.iterdir():
            if not validate_file.is_file():
                continue
            test_file = test_dir / validate_file.name.replace("validate_", "test_", 1)
            try:
                os.link(validate_file, test_file)
            except OSError:
                shutil.copy2(validate_file, test_file)


def _digits_base_manifest(*, shard_paths: list[str], train_examples: int, validate_examples: int) -> dict:
    return {
        "version": DIGITS_DENSE_MANIFEST_VERSION,
        "dataset": "mnist_digits",
        "url_base": DIGITS_DENSE_URL_BASE,
        "dtype": "fp16",
        "layout": "flat_28x28",
        "normalization": "uint8_div_255",
        "num_classes": DIGITS_NUM_CLASSES,
        "num_shards": DIGITS_DENSE_NUM_SHARDS,
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": validate_examples,
        "example_shape": [DIGITS_INPUT_FEATURES],
        "image_shape": [1, DIGITS_IMAGE_HEIGHT, DIGITS_IMAGE_WIDTH],
        "label_shape": [DIGITS_NUM_CLASSES],
        "shard_paths": sorted(str(Path(path)) for path in shard_paths),
        "label_names": [_class_dir(label) for label in range(DIGITS_NUM_CLASSES)],
    }


def _read_digits_manifest_if_valid(cache_root: Path):
    manifest_file = _digits_manifest_path(cache_root)
    if DIGITS_DENSE_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None

    expected = {
        "version": DIGITS_DENSE_MANIFEST_VERSION,
        "dataset": "mnist_digits",
        "url_base": DIGITS_DENSE_URL_BASE,
        "dtype": "fp16",
        "layout": "flat_28x28",
        "normalization": "uint8_div_255",
        "num_classes": DIGITS_NUM_CLASSES,
        "num_shards": DIGITS_DENSE_NUM_SHARDS,
        "example_shape": [DIGITS_INPUT_FEATURES],
        "label_shape": [DIGITS_NUM_CLASSES],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    shard_paths = manifest.get("shard_paths")
    if not shard_paths:
        return None
    if not all(Path(path).exists() for path in shard_paths):
        return None
    return manifest


def _ensure_digits_dense_shards():
    DIGITS_DENSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_digits_manifest_if_valid(DIGITS_DENSE_CACHE_DIR)
    if manifest is not None:
        return manifest

    downloads = _ensure_mnist_downloads(DIGITS_DENSE_CACHE_DIR)
    train_images = _read_idx_images(downloads["train_images"])
    train_labels = _read_idx_labels(downloads["train_labels"])
    validate_images = _read_idx_images(downloads["validate_images"])
    validate_labels = _read_idx_labels(downloads["validate_labels"])

    processing_root = DIGITS_DENSE_CACHE_DIR / "processing_tmp"
    raw_root = processing_root / "raw_fp16_flat"
    shard_root = _digits_shard_root(DIGITS_DENSE_CACHE_DIR)
    base_name = "mnist_digits_dense_fp16_flat"

    if processing_root.exists():
        shutil.rmtree(processing_root)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_dest_dirs = []
    for shard_index in range(DIGITS_DENSE_NUM_SHARDS):
        dest = shard_root / f"dest_{shard_index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs.append(dest)

    train_count = _write_digits_split(train_images, train_labels, split_name="train", raw_root=raw_root)
    validate_count = _write_digits_split(validate_images, validate_labels, split_name="validate", raw_root=raw_root)
    _mirror_validate_as_test(raw_root)

    example_size_in_bytes = DIGITS_INPUT_FEATURES * np.dtype(np.float16).itemsize
    shard_paths = thor.training.create_sharded_raw_dataset(
        [str(raw_root)],
        [str(path) for path in shard_dest_dirs],
        base_name,
        example_size_in_bytes,
        thor.DataType.fp16,
    )
    shard_paths = sorted(str(Path(path)) for path in shard_paths)
    for path in shard_paths:
        assert Path(path).exists(), f"expected shard file {path} to exist"

    manifest = _digits_base_manifest(
        shard_paths=shard_paths,
        train_examples=train_count,
        validate_examples=validate_count,
    )
    _digits_manifest_path(DIGITS_DENSE_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(processing_root)
    return manifest


def _digits_dense_loader(*, batch_size: int):
    manifest = _ensure_digits_dense_shards()
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.fp16,
        manifest["label_shape"],
        thor.DataType.fp16,
        batch_size=batch_size,
        dataset_name="mnist_digits_dense_fp16_flat",
    )
    return loader, manifest


def _build_deep_dense_digits_classifier(
    name: str,
    *,
    num_classes: int,
    width: int = DIGITS_DENSE_WIDTH,
    hidden_layers: int = DIGITS_DENSE_HIDDEN_LAYERS,
    dtype=thor.DataType.fp16,
):
    assert width > 0
    assert hidden_layers > 0
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [DIGITS_INPUT_FEATURES], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)

    x = examples.get_feature_output()
    for _ in range(hidden_layers):
        dense = thor.layers.FullyConnected(network, x, width, True, activation=thor.activations.Relu())
        x = dense.get_feature_output()

    logits = thor.layers.FullyConnected(network, x, num_classes, True, activation=None)
    loss = thor.losses.CategoricalCrossEntropy(
        network,
        logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def test_queued_trainer_trains_really_large_deep_fp16_dense_digits_network(capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _digits_dense_loader(batch_size=DIGITS_DENSE_BATCH_SIZE)
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "DIGITS train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "DIGITS validate split unexpectedly has zero batches"
        assert manifest["num_classes"] == manifest["label_shape"][0]
        assert manifest["example_shape"] == [DIGITS_INPUT_FEATURES]

        network = _build_deep_dense_digits_classifier(
            "python_integration_really_large_deep_dense_fp16_digits",
            num_classes=manifest["num_classes"],
        )
        optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.7, decay=0.01)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats=True,
            stats_interval_s=DIGITS_DENSE_STATS_INTERVAL_S,
            max_in_flight_batches=DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=DIGITS_DENSE_EPOCHS)
        _assert_finite_positive_losses_and_flops(stats, model_name="really_large_deep_dense_fp16_digits")
