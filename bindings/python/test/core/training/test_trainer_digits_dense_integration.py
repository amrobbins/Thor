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
from integration_flags import integration_flag_enabled, integration_skip_reason


def _digits_dense_network_dtype_from_env(env_name: str, raw_value: str):
    dtype_name = raw_value.strip().lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return "fp16", thor.DataType.fp16
    if dtype_name in {"fp32", "float32", "single"}:
        return "fp32", thor.DataType.fp32
    raise RuntimeError(f"{env_name} must be one of fp16 or fp32, got {raw_value!r}")


RUN_DIGITS_DENSE_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_DIGITS_DENSE_INTEGRATION")
RUN_DIGITS_DENSE_CV5_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_DIGITS_DENSE_CV5_INTEGRATION")
DIGITS_DENSE_NETWORK_DTYPE_NAME, DIGITS_DENSE_NETWORK_DTYPE = _digits_dense_network_dtype_from_env(
    "THOR_DIGITS_DENSE_NETWORK_DTYPE",
    os.environ.get("THOR_DIGITS_DENSE_NETWORK_DTYPE", "fp16"),
)
DIGITS_DENSE_CV5_NETWORK_DTYPE_NAME, DIGITS_DENSE_CV5_NETWORK_DTYPE = _digits_dense_network_dtype_from_env(
    "THOR_DIGITS_DENSE_CV5_NETWORK_DTYPE",
    os.environ.get("THOR_DIGITS_DENSE_CV5_NETWORK_DTYPE", DIGITS_DENSE_NETWORK_DTYPE_NAME),
)
DIGITS_DENSE_CACHE_DIR = Path(os.environ.get("THOR_DIGITS_DENSE_CACHE_DIR", "/tmp/thor_digits_dense_training"))
DIGITS_DENSE_URL_BASE = os.environ.get(
    "THOR_DIGITS_DENSE_URL_BASE", "https://storage.googleapis.com/cvdf-datasets/mnist")
DIGITS_DENSE_BATCH_SIZE = int(os.environ.get("THOR_DIGITS_DENSE_BATCH_SIZE", "1024"))
DIGITS_DENSE_EPOCHS = int(os.environ.get("THOR_DIGITS_DENSE_EPOCHS", "1"))
DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES = int(os.environ.get("THOR_DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES", "8"))
DIGITS_DENSE_LOADER_QUEUE_DEPTH = int(
    os.environ.get("THOR_DIGITS_DENSE_LOADER_QUEUE_DEPTH", str(max(32, 2 * DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES))))
DIGITS_DENSE_STATS_INTERVAL_S = float(os.environ.get("THOR_DIGITS_DENSE_STATS_INTERVAL_S", "0.0"))
DIGITS_DENSE_REBUILD = os.environ.get("THOR_DIGITS_DENSE_REBUILD") == "1"
DIGITS_DENSE_NUM_SHARDS = int(os.environ.get("THOR_DIGITS_DENSE_NUM_SHARDS", "1"))
DIGITS_DENSE_WIDTH = int(os.environ.get("THOR_DIGITS_DENSE_WIDTH", "8192"))
DIGITS_DENSE_HIDDEN_LAYERS = int(os.environ.get("THOR_DIGITS_DENSE_HIDDEN_LAYERS", "8"))
DIGITS_DENSE_STATS_COLOR = os.environ.get("THOR_DIGITS_DENSE_STATS_COLOR", "auto").lower()
assert DIGITS_DENSE_STATS_COLOR in {"always", "auto", "never"}
DIGITS_DENSE_CV5_BATCH_SIZE = int(os.environ.get("THOR_DIGITS_DENSE_CV5_BATCH_SIZE", "512"))
DIGITS_DENSE_CV5_EPOCHS = int(os.environ.get("THOR_DIGITS_DENSE_CV5_EPOCHS", "1"))
DIGITS_DENSE_CV5_MAX_IN_FLIGHT_BATCHES = int(
    os.environ.get("THOR_DIGITS_DENSE_CV5_MAX_IN_FLIGHT_BATCHES", str(DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES)))
DIGITS_DENSE_CV5_LOADER_QUEUE_DEPTH = int(
    os.environ.get(
        "THOR_DIGITS_DENSE_CV5_LOADER_QUEUE_DEPTH", str(max(32, 2 * DIGITS_DENSE_CV5_MAX_IN_FLIGHT_BATCHES))))
DIGITS_DENSE_CV5_STATS_INTERVAL_S = float(
    os.environ.get("THOR_DIGITS_DENSE_CV5_STATS_INTERVAL_S", str(DIGITS_DENSE_STATS_INTERVAL_S)))
DIGITS_DENSE_CV5_STATS_COLOR = os.environ.get("THOR_DIGITS_DENSE_CV5_STATS_COLOR", DIGITS_DENSE_STATS_COLOR).lower()
assert DIGITS_DENSE_CV5_STATS_COLOR in {"always", "auto", "never"}
DIGITS_DENSE_CV5_SUMMARY_LOGS_PER_SECOND = float(os.environ.get("THOR_DIGITS_DENSE_CV5_SUMMARY_LOGS_PER_SECOND", "2.0"))
DIGITS_DENSE_CV5_WIDTH = int(os.environ.get("THOR_DIGITS_DENSE_CV5_WIDTH", "1024"))
DIGITS_DENSE_CV5_HIDDEN_LAYERS = int(os.environ.get("THOR_DIGITS_DENSE_CV5_HIDDEN_LAYERS", "3"))
DIGITS_DENSE_CV5_ALT_WIDTH = int(
    os.environ.get("THOR_DIGITS_DENSE_CV5_ALT_WIDTH", str(max(16, DIGITS_DENSE_CV5_WIDTH // 2))))
DIGITS_DENSE_CV5_ALT_HIDDEN_LAYERS = int(
    os.environ.get("THOR_DIGITS_DENSE_CV5_ALT_HIDDEN_LAYERS", str(max(1, DIGITS_DENSE_CV5_HIDDEN_LAYERS - 1))))
DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS_RAW = os.environ.get("THOR_DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS")
DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS = (
    None if DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS_RAW in {None, "", "none", "None"} else
    int(DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS_RAW))
DIGITS_DENSE_CV5_MODEL_ARTIFACTS_DIR = Path(
    os.environ.get(
        "THOR_DIGITS_DENSE_CV5_MODEL_ARTIFACTS_DIR",
        str(Path(tempfile.gettempdir()) / "thor_digits_dense_training_runs_cv5_model_artifacts"),
    ))
# Bump whenever the on-disk raw shard format changes so stale /tmp caches are rebuilt.
DIGITS_DENSE_MANIFEST_VERSION = 2
DIGITS_DENSE_CV5_MANIFEST_VERSION = 2
DIGITS_DENSE_CV5_HOLDOUT_TEST_FRACTION = 0.10
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
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+).*?"
    r"flops/s=\s*(?P<flops>[-+0-9.eE]+[KMGTPE]?)")
_RUN_STATUS_RE = re.compile(
    r"INFO runs\[(?P<run>[^\]|]+)(?:\|[^\]]+)?\]:.*\bstatus=(?P<status>completed|failed|cancelled|interrupted|oom|running|starting|not_started)\b"
)


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)


class _NativeOutputTee:
    """Mirror native stdout/stderr immediately while keeping text for assertions."""

    def __init__(self):
        self._saved_fds = {}
        self._tee_processes = []
        self._capture_paths = []
        self._saved_force_color = None
        self._had_force_color = False
        self._set_force_color_for_tty_tee = False

    def __enter__(self):
        _flush_native_stdio_for_capture()
        self._had_force_color = "FORCE_COLOR" in os.environ
        self._saved_force_color = os.environ.get("FORCE_COLOR")
        self._set_force_color_for_tty_tee = False
        if os.isatty(1) and not os.environ.get("NO_COLOR"):
            # Native stdout/stderr are about to be redirected to pipes so the
            # helper tee process can mirror output and capture it for assertions.
            # Preserve color=auto terminal behavior by forcing color only when
            # the original stdout was a TTY. Shell redirection still leaves this
            # unset, so redirected files stay plain.
            os.environ["FORCE_COLOR"] = "1"
            self._set_force_color_for_tty_tee = True
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

            if self._set_force_color_for_tty_tee:
                if self._had_force_color:
                    os.environ["FORCE_COLOR"] = self._saved_force_color
                else:
                    os.environ.pop("FORCE_COLOR", None)

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


def _expects_color_for_stats_color_mode(mode: str) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("CLICOLOR_FORCE") not in {None, "", "0"}:
        return True
    if os.environ.get("FORCE_COLOR") not in {None, "", "0"}:
        return True
    return os.isatty(1)


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
    return cache_root / "shards_raw_v2_fp16_flat"


def _digits_cv5_manifest_path(cache_root: Path) -> Path:
    return cache_root / "mnist_digits_dense_fp16_cv5_manifest.json"


def _digits_cv5_shard_root(cache_root: Path) -> Path:
    return cache_root / "cv5_shards_raw_v2_fp16_flat"


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


def _digits_cv5_base_manifest(*, source_examples: int, cv_examples: int, test_examples: int, folds: list[dict]) -> dict:
    return {
        "version": DIGITS_DENSE_CV5_MANIFEST_VERSION,
        "source_version": DIGITS_DENSE_MANIFEST_VERSION,
        "dataset": "mnist_digits_cv5",
        "url_base": DIGITS_DENSE_URL_BASE,
        "dtype": "fp16",
        "layout": "flat_28x28",
        "normalization": "uint8_div_255",
        "num_classes": DIGITS_NUM_CLASSES,
        "num_folds": 5,
        "holdout_test_fraction": DIGITS_DENSE_CV5_HOLDOUT_TEST_FRACTION,
        "num_shards": DIGITS_DENSE_NUM_SHARDS,
        "source_examples": source_examples,
        "cv_examples": cv_examples,
        "test_examples": test_examples,
        "example_shape": [DIGITS_INPUT_FEATURES],
        "image_shape": [1, DIGITS_IMAGE_HEIGHT, DIGITS_IMAGE_WIDTH],
        "label_shape": [DIGITS_NUM_CLASSES],
        "label_names": [_class_dir(label) for label in range(DIGITS_NUM_CLASSES)],
        "folds": folds,
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


def _read_digits_cv5_manifest_if_valid(cache_root: Path):
    manifest_file = _digits_cv5_manifest_path(cache_root)
    if DIGITS_DENSE_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None

    expected = {
        "version": DIGITS_DENSE_CV5_MANIFEST_VERSION,
        "source_version": DIGITS_DENSE_MANIFEST_VERSION,
        "dataset": "mnist_digits_cv5",
        "url_base": DIGITS_DENSE_URL_BASE,
        "dtype": "fp16",
        "layout": "flat_28x28",
        "normalization": "uint8_div_255",
        "num_classes": DIGITS_NUM_CLASSES,
        "num_folds": 5,
        "holdout_test_fraction": DIGITS_DENSE_CV5_HOLDOUT_TEST_FRACTION,
        "num_shards": DIGITS_DENSE_NUM_SHARDS,
        "example_shape": [DIGITS_INPUT_FEATURES],
        "label_shape": [DIGITS_NUM_CLASSES],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
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
    for fold in folds:
        shard_paths = fold.get("shard_paths") if isinstance(fold, dict) else None
        if not shard_paths or not all(Path(path).exists() for path in shard_paths):
            return None
        if fold.get("test_examples") != test_examples:
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


def _stratified_fold_indices(labels: np.ndarray, *, num_folds: int) -> list[np.ndarray]:
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(num_folds)]
    for label in range(DIGITS_NUM_CLASSES):
        label_indices = np.flatnonzero(labels == label)
        for fold_index in range(num_folds):
            fold_parts[fold_index].append(label_indices[fold_index::num_folds])
    folds = []
    for fold_index in range(num_folds):
        fold_indices = np.concatenate(fold_parts[fold_index])
        fold_indices.sort()
        folds.append(fold_indices)
    return folds


def _stratified_holdout_indices(labels: np.ndarray, *, fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise RuntimeError(f"holdout fraction must be between 0 and 1, got {fraction}")

    total_count = int(labels.shape[0])
    target_count = int(round(total_count * fraction))
    if target_count <= 0 or target_count >= total_count:
        raise RuntimeError(f"invalid holdout target count {target_count} for {total_count} examples")

    label_parts = []
    allocated = 0
    for label in range(DIGITS_NUM_CLASSES):
        label_indices = np.flatnonzero(labels == label)
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
        label_indices = label_part["indices"]
        # Use a deterministic strided sample from each class instead of taking
        # one contiguous prefix of the IDX order. This keeps the hold-out set
        # representative without adding random state to the integration cache.
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


def _ensure_digits_dense_cv5_shards():
    DIGITS_DENSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_digits_cv5_manifest_if_valid(DIGITS_DENSE_CACHE_DIR)
    if manifest is not None:
        return manifest

    downloads = _ensure_mnist_downloads(DIGITS_DENSE_CACHE_DIR)
    train_images = _read_idx_images(downloads["train_images"])
    train_labels = _read_idx_labels(downloads["train_labels"])

    processing_root = DIGITS_DENSE_CACHE_DIR / "cv5_processing_tmp"
    raw_root = processing_root / "raw_fp16_flat"
    shard_root = _digits_cv5_shard_root(DIGITS_DENSE_CACHE_DIR)
    base_name = "mnist_digits_dense_fp16_cv5"

    if processing_root.exists():
        shutil.rmtree(processing_root)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    all_indices = np.arange(train_labels.shape[0])
    holdout_indices = _stratified_holdout_indices(
        train_labels,
        fraction=DIGITS_DENSE_CV5_HOLDOUT_TEST_FRACTION,
    )
    cv_mask = np.ones(train_labels.shape[0], dtype=bool)
    cv_mask[holdout_indices] = False
    cv_indices = all_indices[cv_mask]
    fold_indices = _stratified_fold_indices(train_labels[cv_indices], num_folds=5)
    example_size_in_bytes = DIGITS_INPUT_FEATURES * np.dtype(np.float16).itemsize
    folds = []

    for fold_index, validate_relative_indices in enumerate(fold_indices):
        fold_cv_train_mask = np.ones(cv_indices.shape[0], dtype=bool)
        fold_cv_train_mask[validate_relative_indices] = False
        train_indices = cv_indices[fold_cv_train_mask]
        validate_indices = cv_indices[validate_relative_indices]

        fold_raw_root = raw_root / f"fold_{fold_index}"
        fold_shard_root = shard_root / f"fold_{fold_index}"
        fold_shard_root.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs = []
        for shard_index in range(DIGITS_DENSE_NUM_SHARDS):
            dest = fold_shard_root / f"dest_{shard_index:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            shard_dest_dirs.append(dest)

        train_count = _write_digits_split(
            train_images[train_indices], train_labels[train_indices], split_name="train", raw_root=fold_raw_root)
        validate_count = _write_digits_split(
            train_images[validate_indices],
            train_labels[validate_indices],
            split_name="validate",
            raw_root=fold_raw_root)
        test_count = _write_digits_split(
            train_images[holdout_indices], train_labels[holdout_indices], split_name="test", raw_root=fold_raw_root)

        shard_paths = thor.training.create_sharded_raw_dataset(
            [str(fold_raw_root)],
            [str(path) for path in shard_dest_dirs],
            f"{base_name}_fold_{fold_index}",
            example_size_in_bytes,
            thor.DataType.fp16,
        )
        shard_paths = sorted(str(Path(path)) for path in shard_paths)
        for path in shard_paths:
            assert Path(path).exists(), f"expected CV shard file {path} to exist"

        folds.append(
            {
                "fold_index": fold_index,
                "train_examples": train_count,
                "validate_examples": validate_count,
                "test_examples": test_count,
                "shard_paths": shard_paths,
            })

    manifest = _digits_cv5_base_manifest(
        source_examples=int(train_labels.shape[0]),
        cv_examples=int(cv_indices.shape[0]),
        test_examples=int(holdout_indices.shape[0]),
        folds=folds,
    )
    _digits_cv5_manifest_path(DIGITS_DENSE_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(processing_root)
    return manifest


def _digits_dense_loader_from_manifest(manifest: dict, *, batch_size: int, batch_queue_depth: int, dataset_name: str):
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.fp16,
        manifest["label_shape"],
        thor.DataType.fp16,
        batch_size=batch_size,
        dataset_name=dataset_name,
        batch_queue_depth=batch_queue_depth,
    )
    return loader


def _digits_dense_loader(*, batch_size: int):
    manifest = _ensure_digits_dense_shards()
    loader = _digits_dense_loader_from_manifest(
        manifest,
        batch_size=batch_size,
        batch_queue_depth=DIGITS_DENSE_LOADER_QUEUE_DEPTH,
        dataset_name="mnist_digits_dense_fp16_flat",
    )
    return loader, manifest


def _build_deep_dense_digits_classifier(
    name: str,
    *,
    num_classes: int,
    width: int = DIGITS_DENSE_WIDTH,
    hidden_layers: int = DIGITS_DENSE_HIDDEN_LAYERS,
    dtype=DIGITS_DENSE_NETWORK_DTYPE,
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


@pytest.mark.skipif(
    not RUN_DIGITS_DENSE_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_DIGITS_DENSE_INTEGRATION",
        description="the heavyweight fp16 dense DIGITS/MNIST training throughput test",
    ),
)
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
            f"python_integration_really_large_deep_dense_{DIGITS_DENSE_NETWORK_DTYPE_NAME}_digits",
            num_classes=manifest["num_classes"],
            dtype=DIGITS_DENSE_NETWORK_DTYPE,
        )
        optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.7, decay=0.01)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats_interval_s=DIGITS_DENSE_STATS_INTERVAL_S,
            max_in_flight_batches=DIGITS_DENSE_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color=DIGITS_DENSE_STATS_COLOR,
        )
        stats = _fit_and_capture_stats(trainer, epochs=DIGITS_DENSE_EPOCHS)
        _assert_finite_positive_losses_and_flops(stats, model_name="really_large_deep_dense_fp16_digits")


@pytest.mark.digits_dense_cv5_integration
@pytest.mark.skipif(
    not RUN_DIGITS_DENSE_CV5_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_DIGITS_DENSE_CV5_INTEGRATION",
        description="the DIGITS/MNIST dense 5-fold CV TrainingRuns test",
    ),
)
def test_training_runs_digits_dense_five_fold_cross_validation(capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        cv_manifest = _ensure_digits_dense_cv5_shards()
        assert cv_manifest["num_folds"] == 5
        assert len(cv_manifest["folds"]) == 5
        assert cv_manifest["example_shape"] == [DIGITS_INPUT_FEATURES]
        assert cv_manifest["label_shape"] == [DIGITS_NUM_CLASSES]
        assert cv_manifest["test_examples"] == int(
            round(cv_manifest["source_examples"] * DIGITS_DENSE_CV5_HOLDOUT_TEST_FRACTION))
        assert cv_manifest["cv_examples"] + cv_manifest["test_examples"] == cv_manifest["source_examples"]

        def make_fold_trainer(
            *,
            fold: dict,
            run_name: str,
            model_name: str,
            width: int,
            hidden_layers: int,
            save_model_dir: Path,
        ):
            fold_index = int(fold["fold_index"])
            fold_manifest = {
                **cv_manifest,
                "shard_paths": fold["shard_paths"],
                "train_examples": fold["train_examples"],
                "validate_examples": fold["validate_examples"],
                "test_examples": fold["test_examples"],
            }
            loader = _digits_dense_loader_from_manifest(
                fold_manifest,
                batch_size=DIGITS_DENSE_CV5_BATCH_SIZE,
                batch_queue_depth=DIGITS_DENSE_CV5_LOADER_QUEUE_DEPTH,
                dataset_name=f"mnist_digits_dense_fp16_cv5_{run_name}",
            )
            assert fold["train_examples"] == loader.get_num_train_examples()
            assert fold["validate_examples"] == loader.get_num_validate_examples()
            assert fold["test_examples"] == cv_manifest["test_examples"]
            assert loader.get_num_train_batches() > 0
            assert loader.get_num_validate_batches() > 0

            network = _build_deep_dense_digits_classifier(
                model_name,
                num_classes=cv_manifest["num_classes"],
                width=width,
                hidden_layers=hidden_layers,
                dtype=DIGITS_DENSE_CV5_NETWORK_DTYPE,
            )
            #optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.7, decay=0.01)
            optimizer = thor.optimizers.Adam()
            return thor.training.Trainer(
                network,
                loader,
                optimizer=optimizer,
                debug_synchronous=False,
                stats_interval_s=DIGITS_DENSE_CV5_STATS_INTERVAL_S,
                max_in_flight_batches=DIGITS_DENSE_CV5_MAX_IN_FLIGHT_BATCHES,
                scalar_tensors_to_report=["loss"],
                stats_color=DIGITS_DENSE_CV5_STATS_COLOR,
                save_model_dir=str(save_model_dir),
                save_model_overwrite=True,
            )

        artifact_root = DIGITS_DENSE_CV5_MODEL_ARTIFACTS_DIR
        run_specs = []
        for fold in cv_manifest["folds"]:
            fold_index = int(fold["fold_index"])
            trainer = make_fold_trainer(
                fold=fold,
                run_name=f"fold_{fold_index}",
                model_name=f"python_integration_digits_dense_cv5_fold_{fold_index}",
                width=DIGITS_DENSE_CV5_WIDTH,
                hidden_layers=DIGITS_DENSE_CV5_HIDDEN_LAYERS,
                save_model_dir=artifact_root / f"fold_{fold_index}",
            )
            run_specs.append((f"fold_{fold_index}", trainer, "digits_dense_cv5"))

        # for fold in cv_manifest["folds"][:3]:
        #     fold_index = int(fold["fold_index"])
        #     trainer = make_fold_trainer(
        #         fold=fold,
        #         run_name=f"alt_fold_{fold_index}",
        #         model_name=f"python_integration_digits_dense_alt3_fold_{fold_index}",
        #         width=DIGITS_DENSE_CV5_ALT_WIDTH,
        #         hidden_layers=DIGITS_DENSE_CV5_ALT_HIDDEN_LAYERS,
        #         save_model_dir=artifact_root / f"alt_fold_{fold_index}",
        #     )
        #     run_specs.append((f"alt_fold_{fold_index}", trainer, "digits_dense_alt3"))

        runs = thor.training.TrainingRuns(
            run_specs,
            max_summary_logs_per_second=DIGITS_DENSE_CV5_SUMMARY_LOGS_PER_SECOND,
            max_parallel_runs=DIGITS_DENSE_CV5_MAX_PARALLEL_RUNS,
        )
        test_fold = cv_manifest["folds"][0]
        test_manifest = {
            **cv_manifest,
            "shard_paths": test_fold["shard_paths"],
            "train_examples": test_fold["train_examples"],
            "validate_examples": test_fold["validate_examples"],
            "test_examples": test_fold["test_examples"],
        }
        test_loader = _digits_dense_loader_from_manifest(
            test_manifest,
            batch_size=DIGITS_DENSE_CV5_BATCH_SIZE,
            batch_queue_depth=DIGITS_DENSE_CV5_LOADER_QUEUE_DEPTH,
            dataset_name="mnist_digits_dense_fp16_cv5_holdout_test",
        )
        results, captured_text = _fit_training_runs_and_capture_text(
            runs,
            epochs=DIGITS_DENSE_CV5_EPOCHS,
            test_loader=test_loader,
        )

    plain_text = _ANSI_RE.sub("", captured_text)
    statuses = _captured_run_statuses(captured_text)

    if _expects_color_for_stats_color_mode(DIGITS_DENSE_CV5_STATS_COLOR):
        assert "\x1b[" in captured_text, "TrainingRuns DIGITS CV5 output should preserve ANSI color in color-enabled runs"

    assert len(results) == 5
    assert results.all_completed()
    assert "INFO runs summary:" in plain_text
    assert "\nINFO runs final: ==================== final results" in plain_text
    assert "INFO runs final: total=5" in plain_text
    assert "INFO runs final: =====================================================" in plain_text
    assert "INFO runs ensemble:" in plain_text
    assert "INFO runs ensemble[digits_dense_cv5]:" in plain_text
    # assert "INFO runs ensemble[digits_dense_alt3]:" in plain_text
    assert "aggregation=ensemble_eval" in plain_text
    assert "ensemble_train_loss=" in plain_text
    assert "ensemble_test_loss=" in plain_text
    assert "ensemble_test_accuracy=" in plain_text
    assert "weighted_train_loss=" not in plain_text
    assert "weighted_validate_loss=" not in plain_text
    assert "INFO runs[fold_0|digits_dense_cv5]:" in plain_text
    # assert "INFO runs[alt_fold_0|digits_dense_alt3]:" in plain_text
    assert "ensemble_group=digits_dense_cv5" not in plain_text
    assert "ensemble_group=digits_dense_alt3" not in plain_text
    assert "train_loss=" in plain_text
    assert "validate_loss=" in plain_text
    assert "test_loss=" in plain_text
    for fold_index in range(5):
        run_name = f"fold_{fold_index}"
        assert re.search(
            rf"INFO runs\[{re.escape(run_name)}\|digits_dense_cv5\]:.*train_loss=.*validate_loss=.*test_loss=.*test_accuracy=",
            plain_text,
        ), f"final report did not include per-fold test_loss/test_accuracy for {run_name}:\n{plain_text}"
    assert "completed=5" in plain_text
    assert results.status_counts["completed"] == 5
    assert results.has_ensembles
    assert len(results.ensembles) == 1

    cv5_ensemble = results.ensemble("digits_dense_cv5")
    assert cv5_ensemble.all_completed()
    assert cv5_ensemble.total_weight == pytest.approx(5.0)
    assert len(cv5_ensemble.members) == 5
    assert cv5_ensemble.ensemble_train_loss is not None
    assert cv5_ensemble.ensemble_test_loss is not None
    assert cv5_ensemble.ensemble_test_accuracy is not None

    # alt3_ensemble = results.ensemble("digits_dense_alt3")
    # assert alt3_ensemble.all_completed()
    # assert alt3_ensemble.total_weight == pytest.approx(3.0)
    # assert len(alt3_ensemble.members) == 3
    # assert alt3_ensemble.ensemble_train_loss is not None
    # assert alt3_ensemble.ensemble_test_loss is None
    # assert "phase=unknown" not in plain_text
    # assert "INFO trainer:" not in plain_text

    validation_losses = []
    test_losses = []
    expected_groups = {
        **{
            f"fold_{fold_index}": "digits_dense_cv5" for fold_index in range(5)
        },
        # **{
        #     f"alt_fold_{fold_index}": "digits_dense_alt3" for fold_index in range(3)
        # },
    }
    for run_name, ensemble_group in expected_groups.items():
        result = results[run_name]
        assert statuses[run_name] == "completed"
        assert result.status == "completed"
        assert result.ensemble_group == ensemble_group
        assert result.ensemble_weight == pytest.approx(1.0)
        assert result.final_training_loss is not None
        assert result.final_validation_loss is not None
        assert result.final_test_loss is not None
        assert result.final_test_accuracy is not None
        assert result.final_loss("train") == result.final_training_loss
        assert result.final_loss("validate") == result.final_validation_loss
        assert result.final_loss("test") == result.final_test_loss
        assert result.final_accuracy("test") == result.final_test_accuracy
        assert math.isfinite(result.final_training_loss)
        assert math.isfinite(result.final_validation_loss)
        assert math.isfinite(result.final_test_loss)
        assert math.isfinite(result.final_test_accuracy)
        assert 0.0 <= result.final_test_accuracy <= 1.0
        assert result.final_training_loss > 0.0
        assert result.final_validation_loss > 0.0
        assert result.final_test_loss > 0.0
        assert result.final_training_step is not None
        assert result.final_validation_step is not None
        assert result.final_test_step is not None
        validation_losses.append(result.final_validation_loss)
        test_losses.append(result.final_test_loss)

    mean_validation_loss = float(np.mean(validation_losses))
    assert math.isfinite(mean_validation_loss)
    assert mean_validation_loss > 0.0
    mean_test_loss = float(np.mean(test_losses))
    assert math.isfinite(mean_test_loss)
    assert mean_test_loss > 0.0
    assert math.isfinite(cv5_ensemble.ensemble_test_loss)
    assert cv5_ensemble.ensemble_test_loss > 0.0
    assert math.isfinite(cv5_ensemble.ensemble_test_accuracy)
    assert 0.0 <= cv5_ensemble.ensemble_test_accuracy <= 1.0
