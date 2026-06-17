import concurrent.futures
import ctypes
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
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import thor

RUN_BYTE_LM_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_BYTE_LM_INTEGRATION") == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


BYTE_LM_CACHE_DIR = Path(os.environ.get("THOR_BYTE_LM_CACHE_DIR", "/tmp/thor_byte_lm_training"))
BYTE_LM_DATASET_BASE_URL = os.environ.get(
    "THOR_BYTE_LM_DATASET_BASE_URL",
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT",
)
BYTE_LM_DATASET_NAME = os.environ.get("THOR_BYTE_LM_DATASET_NAME", "HuggingFaceFW/fineweb-edu/sample-100BT")
BYTE_LM_LOCAL_DATASET_DIR = Path(
    os.environ.get("THOR_BYTE_LM_LOCAL_DATASET_DIR", str(Path.home() / "fineweb-edu-100BT"))
)
BYTE_LM_CONTEXT_LENGTH = _env_int("THOR_BYTE_LM_CONTEXT_LENGTH", 256)
BYTE_LM_STRIDE = _env_int("THOR_BYTE_LM_STRIDE", BYTE_LM_CONTEXT_LENGTH)
BYTE_LM_BATCH_SIZE = _env_int("THOR_BYTE_LM_BATCH_SIZE", 32)
BYTE_LM_EPOCHS = _env_int("THOR_BYTE_LM_EPOCHS", 1)
BYTE_LM_MAX_IN_FLIGHT_BATCHES = _env_int("THOR_BYTE_LM_MAX_IN_FLIGHT_BATCHES", 8)
BYTE_LM_LOADER_QUEUE_DEPTH = int(
    os.environ.get("THOR_BYTE_LM_LOADER_QUEUE_DEPTH", str(max(32, 2 * BYTE_LM_MAX_IN_FLIGHT_BATCHES)))
)
BYTE_LM_REQUEST_FULL_DATASET = os.environ.get("THOR_BYTE_LM_TARGET_TEXT_BYTES") == "0"
BYTE_LM_TRAIN_EXAMPLES = _env_int("THOR_BYTE_LM_TRAIN_EXAMPLES", 0 if BYTE_LM_REQUEST_FULL_DATASET else 131072)
BYTE_LM_VALIDATE_EXAMPLES = _env_int("THOR_BYTE_LM_VALIDATE_EXAMPLES", 0 if BYTE_LM_REQUEST_FULL_DATASET else 4096)
BYTE_LM_TEST_EXAMPLES = _env_int("THOR_BYTE_LM_TEST_EXAMPLES", 0 if BYTE_LM_REQUEST_FULL_DATASET else 4096)
BYTE_LM_TARGET_TEXT_BYTES = _env_int(
    "THOR_BYTE_LM_TARGET_TEXT_BYTES",
    max(
        1,
        (BYTE_LM_TRAIN_EXAMPLES + BYTE_LM_VALIDATE_EXAMPLES + BYTE_LM_TEST_EXAMPLES) * BYTE_LM_STRIDE
        + 3 * (BYTE_LM_CONTEXT_LENGTH + 1),
    ),
)
BYTE_LM_NUM_SHARDS = _env_int("THOR_BYTE_LM_NUM_SHARDS", 5 if BYTE_LM_TARGET_TEXT_BYTES == 0 else 1)
BYTE_LM_MAX_PARQUET_FILES = _env_int("THOR_BYTE_LM_MAX_PARQUET_FILES", 0 if BYTE_LM_TARGET_TEXT_BYTES == 0 else 64)
BYTE_LM_HIDDEN_DIM = _env_int("THOR_BYTE_LM_HIDDEN_DIM", 512)
BYTE_LM_NUM_HEADS = _env_int("THOR_BYTE_LM_NUM_HEADS", 8)
BYTE_LM_NUM_LAYERS = _env_int("THOR_BYTE_LM_NUM_LAYERS", 8)
BYTE_LM_STATS_INTERVAL_S = float(os.environ.get("THOR_BYTE_LM_STATS_INTERVAL_S", "0.0"))
BYTE_LM_STATS_COLOR = os.environ.get("THOR_BYTE_LM_STATS_COLOR", "auto").lower()
BYTE_LM_LEARNING_RATE = float(os.environ.get("THOR_BYTE_LM_LEARNING_RATE", "3.0e-4"))
BYTE_LM_WEIGHT_DECAY = float(os.environ.get("THOR_BYTE_LM_WEIGHT_DECAY", "0.1"))
BYTE_LM_LOSS_WEIGHT = float(os.environ.get("THOR_BYTE_LM_LOSS_WEIGHT", str(1.0 / BYTE_LM_CONTEXT_LENGTH)))
BYTE_LM_SAVE_DIR_ENV = os.environ.get("THOR_BYTE_LM_SAVE_DIR")
BYTE_LM_SAVE_DIR = Path(BYTE_LM_SAVE_DIR_ENV) if BYTE_LM_SAVE_DIR_ENV else None
BYTE_LM_SAVE_OVERWRITE = os.environ.get("THOR_BYTE_LM_SAVE_OVERWRITE", "1") == "1"
BYTE_LM_SAVE_OPTIMIZER_STATE = os.environ.get("THOR_BYTE_LM_SAVE_OPTIMIZER_STATE", "1") == "1"
BYTE_LM_REBUILD = os.environ.get("THOR_BYTE_LM_REBUILD") == "1"
BYTE_LM_VOCAB_SIZE = 256
BYTE_LM_FULL_DATASET = BYTE_LM_TARGET_TEXT_BYTES == 0
BYTE_LM_SHARD_STRIPE_BYTES = _env_int("THOR_BYTE_LM_SHARD_STRIPE_BYTES", 4 * 1024 * 1024 * 1024)
BYTE_LM_CACHE_IO_BUFFER_BYTES = _env_int("THOR_BYTE_LM_CACHE_IO_BUFFER_BYTES", 64 * 1024 * 1024)
BYTE_LM_MAX_PARQUET_WORKERS = max(1, (os.cpu_count() or 2) - 1)
BYTE_LM_PARQUET_WORKERS = min(
    BYTE_LM_MAX_PARQUET_WORKERS,
    max(1, _env_int("THOR_BYTE_LM_PARQUET_WORKERS", BYTE_LM_MAX_PARQUET_WORKERS)),
)
# Pre-alpha development intentionally keeps cache/shard versions at 1. Delete/rebuild caches when the layout changes.
BYTE_LM_MANIFEST_VERSION = 1

assert BYTE_LM_STATS_COLOR in {"always", "auto", "never"}

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.byte_lm_integration,
    pytest.mark.skipif(
        not RUN_BYTE_LM_INTEGRATION,
        reason=(
            "set THOR_RUN_TRAINING_BYTE_LM_INTEGRATION=1 to run the heavyweight "
            "FineWeb-Edu byte-level causal LM training test"
        ),
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+).*?"
    r"flops/s=\s*(?P<flops>[-+0-9.eE]+[KMGTPE]?)"
)


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)


class _NativeOutputTee:
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
        if os.isatty(1) and not os.environ.get("NO_COLOR"):
            os.environ["FORCE_COLOR"] = "1"
            self._set_force_color_for_tty_tee = True
        tee_exe = shutil.which("tee")
        assert tee_exe is not None, "the temporary native-output tee requires /usr/bin/tee on PATH"

        for fd in (1, 2):
            saved_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            capture_file = tempfile.NamedTemporaryFile(prefix=f"thor_byte_lm_fit_fd{fd}_", suffix=".log", delete=False)
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
            }
        )
    return stats


def _fit_and_capture_stats(trainer, *, epochs: int):
    with _NativeOutputTee() as tee:
        trainer.fit(epochs=epochs)
    stats = _captured_trainer_stats(tee.text())
    assert stats, "trainer emitted no parseable stats; lower THOR_BYTE_LM_STATS_INTERVAL_S if this happens"
    return stats


def _assert_finite_positive_losses(stats, *, model_name: str):
    assert any(entry["phase"] == "train" for entry in stats), f"{model_name}: no train stats reported"
    assert any(entry["phase"] == "validate" for entry in stats), f"{model_name}: no validate stats reported"
    for entry in stats:
        loss = entry["loss"]
        assert math.isfinite(loss), f"{model_name}: non-finite loss reported: {entry}"
        assert loss > 0.0, f"{model_name}: non-positive loss reported: {entry}"



class _ResidualAdd(thor.layers.CustomLayer):
    def __init__(self, network: thor.Network, lhs: thor.Tensor, rhs: thor.Tensor):
        assert lhs.get_dimensions() == rhs.get_dimensions()
        assert lhs.get_data_type() == rhs.get_data_type()

        def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            return {"feature_output": context.input("lhs") + context.input("rhs")}

        super().__init__(
            network=network,
            inputs={"lhs": lhs, "rhs": rhs},
            output_names=["feature_output"],
            build=build,
            parameters=[],
        )


def _flops_value(raw_value: str) -> float:
    suffix_scale = {"": 1.0, "K": 1.0e3, "M": 1.0e6, "G": 1.0e9, "T": 1.0e12, "P": 1.0e15, "E": 1.0e18}
    suffix = raw_value[-1]
    if suffix.isalpha():
        return float(raw_value[:-1]) * suffix_scale[suffix]
    return float(raw_value)


def _byte_lm_manifest_path(cache_root: Path) -> Path:
    return cache_root / "fineweb_edu_byte_lm_manifest.json"


def _download_parquet_if_missing(index: int, downloads_root: Path) -> Path:
    filename = f"000_{index:05d}.parquet"
    local_candidates = (
        BYTE_LM_LOCAL_DATASET_DIR / "sample" / "100BT" / filename,
        BYTE_LM_LOCAL_DATASET_DIR / filename,
    )
    for candidate in local_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    path = downloads_root / filename
    if path.exists() and path.stat().st_size > 0:
        return path

    # The high-performance integration path is intentionally local-first: users
    # should download the FineWeb-Edu shard set once with hf_transfer rather than
    # letting the test dribble individual parquet files through urllib.  Keep the
    # remote fallback for convenience, but make the failure actionable and also
    # try huggingface_hub when it is installed because HF dataset file URLs have
    # changed in the past.
    path.parent.mkdir(parents=True, exist_ok=True)
    remote_filename = f"sample/100BT/{filename}"
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            repo_type="dataset",
            filename=remote_filename,
            local_dir=str(downloads_root.parent.parent),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.exists() and downloaded_path.stat().st_size > 0:
            return downloaded_path
    except Exception:
        pass

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    url = f"{BYTE_LM_DATASET_BASE_URL.rstrip('/')}/{filename}"
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception as exc:
        raise RuntimeError(
            "FineWeb-Edu parquet file is not available locally and remote fallback failed. "
            "Download the dataset first with:\n"
            "  python3 -m pip install -U 'huggingface_hub[hf_transfer]'\n"
            "  HF_HUB_ENABLE_HF_TRANSFER=1 hf download HuggingFaceFW/fineweb-edu "
            "--repo-type dataset --include 'sample/100BT/*' "
            "--local-dir '$HOME/fineweb-edu-100BT'\n"
            f"Missing local candidates: {[str(candidate) for candidate in local_candidates]}"
        ) from exc
    tmp_path.replace(path)
    return path



_SHARD_MAGIC = b"THOR_RAW_SHARD\0\0"
_SHARD_FORMAT_VERSION = 1
_SHARD_HEADER_BYTES = 88
_SHARD_UINT8_DTYPE = 17
_SHARD_LAYOUT_BYTE_CORPUS = 2
_DOC_SEPARATOR = b"\n\n"
_SPLIT_NAMES = ("train", "validate", "test")
_FULL_DATASET_TRAIN_PCT = 98
_FULL_DATASET_VALIDATE_PCT = 1
_FULL_DATASET_TEST_PCT = 1


def _hash_split(parts: Sequence[object]) -> str:
    """Deterministically assign a document payload to train/validate/test.

    Full-dataset byte-corpus builds stream whole documents into one split so that
    contiguous byte offsets remain meaningful for next-byte labels.  Hashing the
    UTF-8 payload gives stable splits independent of parquet iteration chunking
    while avoiding a per-document metadata table.
    """

    crc = 0
    for part in parts:
        crc = zlib.crc32(part, crc)
    bucket = crc % 100
    if bucket < _FULL_DATASET_TRAIN_PCT:
        return "train"
    if bucket < _FULL_DATASET_TRAIN_PCT + _FULL_DATASET_VALIDATE_PCT:
        return "validate"
    return "test"


def _payload_size(parts: Sequence[object]) -> int:
    return sum(len(part) for part in parts)


def _write_u64(stream, value: int):
    stream.write(struct.pack("<Q", int(value)))


def _write_string(stream, value: str):
    payload = value.encode("utf-8")
    _write_u64(stream, len(payload))
    stream.write(payload)


def _local_parquet_paths() -> list[Path]:
    roots = [
        BYTE_LM_LOCAL_DATASET_DIR / "sample" / "100BT",
        BYTE_LM_LOCAL_DATASET_DIR,
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def _fineweb_edu_parquet_paths(downloads_root: Path) -> list[Path]:
    local_paths = _local_parquet_paths()
    if local_paths:
        if BYTE_LM_MAX_PARQUET_FILES > 0:
            return local_paths[:BYTE_LM_MAX_PARQUET_FILES]
        return local_paths

    if BYTE_LM_MAX_PARQUET_FILES == 0:
        raise RuntimeError(
            "THOR_BYTE_LM_TARGET_TEXT_BYTES=0 / THOR_BYTE_LM_MAX_PARQUET_FILES=0 requires local parquet files. "
            f"No *.parquet files found under {BYTE_LM_LOCAL_DATASET_DIR}."
        )
    return [_download_parquet_if_missing(index, downloads_root) for index in range(BYTE_LM_MAX_PARQUET_FILES)]



def _byte_corpus_num_examples(num_bytes: int, *, context_length: int, stride: int) -> int:
    if num_bytes <= context_length:
        return 0
    return 1 + (num_bytes - context_length - 1) // stride


class _ByteCorpusAppendWriter:
    def __init__(self, cache_root: Path, *, num_shards: int, context_length: int, stride: int, stripe_bytes: int):
        if num_shards <= 0:
            raise ValueError("THOR_BYTE_LM_NUM_SHARDS must be >= 1")
        if stripe_bytes <= 0:
            raise ValueError("THOR_BYTE_LM_SHARD_STRIPE_BYTES must be >= 1")

        self.cache_root = cache_root
        self.num_shards = num_shards
        self.context_length = context_length
        self.stride = stride
        self.stripe_bytes = stripe_bytes
        self.base_shard_index = 0
        self.shard_root = cache_root / "byte_corpus_shards_v1_uint8_shifted_labels"
        if self.shard_root.exists():
            shutil.rmtree(self.shard_root)
        self.shard_root.mkdir(parents=True, exist_ok=True)

        self.files: dict[tuple[str, int], object] = {}
        self.shard_paths: list[str] = []
        self.byte_counts = {split: [0 for _ in range(num_shards)] for split in _SPLIT_NAMES}
        self.total_bytes = {split: 0 for split in _SPLIT_NAMES}

        for split in _SPLIT_NAMES:
            for shard_index in range(num_shards):
                shard_path = self.shard_root / f"fineweb_edu_byte_corpus_{split}_{shard_index + 1}_of_{num_shards}.shard"
                shard_file = shard_path.open("wb", buffering=BYTE_LM_CACHE_IO_BUFFER_BYTES)
                shard_file.write(b"\0" * _SHARD_HEADER_BYTES)
                self.files[(split, shard_index)] = shard_file
                self.shard_paths.append(str(shard_path))

    def close(self):
        for shard_file in self.files.values():
            shard_file.close()

    def _write_part(self, split: str, part) -> None:
        part_view = memoryview(part)
        offset = 0
        while offset < len(part_view):
            split_written = self.total_bytes[split]
            shard_index = (self.base_shard_index + split_written // self.stripe_bytes) % self.num_shards
            stripe_remaining = self.stripe_bytes - (split_written % self.stripe_bytes)
            chunk_size = min(len(part_view) - offset, stripe_remaining)
            shard_file = self.files[(split, shard_index)]
            shard_file.write(part_view[offset : offset + chunk_size])
            self.byte_counts[split][shard_index] += chunk_size
            self.total_bytes[split] += chunk_size
            offset += chunk_size

    def add_parts(self, split: str, parts: Sequence[object]) -> None:
        for part in parts:
            if len(part) != 0:
                self._write_part(split, part)

    def add_slice(self, split: str, parts: Sequence[object], start: int, size: int) -> None:
        if size <= 0:
            return
        remaining = size
        skip = start
        for part in parts:
            part_len = len(part)
            if skip >= part_len:
                skip -= part_len
                continue
            part_view = memoryview(part)
            take = min(part_len - skip, remaining)
            self._write_part(split, part_view[skip : skip + take])
            remaining -= take
            skip = 0
            if remaining == 0:
                return
        if remaining != 0:
            raise RuntimeError(f"byte corpus slice overran payload by {remaining} bytes")

    def add_shard_part(self, split: str, shard_index: int, part_path: Path) -> None:
        part_bytes = part_path.stat().st_size
        if part_bytes == 0:
            return
        shard_file = self.files[(split, shard_index)]
        with part_path.open("rb", buffering=BYTE_LM_CACHE_IO_BUFFER_BYTES) as source:
            shutil.copyfileobj(source, shard_file, length=BYTE_LM_CACHE_IO_BUFFER_BYTES)
        self.byte_counts[split][shard_index] += part_bytes
        self.total_bytes[split] += part_bytes

    def finalize(self) -> list[str]:
        for (file_split, shard_index), shard_file in self.files.items():
            shard_file.flush()
            metadata_offset = shard_file.tell()
            _write_string(shard_file, "sequence")
            _write_u64(shard_file, self.stride)
            split_counts: dict[str, int] = {}
            for split in _SPLIT_NAMES:
                byte_count = self.byte_counts[split][shard_index] if split == file_split else 0
                example_count = _byte_corpus_num_examples(
                    byte_count,
                    context_length=self.context_length,
                    stride=self.stride,
                )
                split_counts[split] = example_count
                _write_u64(shard_file, _SHARD_HEADER_BYTES)
                _write_u64(shard_file, example_count)
                _write_u64(shard_file, byte_count)
            metadata_end = shard_file.tell()
            _write_byte_corpus_shard_header(
                shard_file,
                record_size=self.context_length + 1,
                train_count=split_counts["train"],
                validate_count=split_counts["validate"],
                test_count=split_counts["test"],
                metadata_offset=metadata_offset,
                metadata_bytes=metadata_end - metadata_offset,
            )
            shard_file.flush()
        return self.shard_paths


class _ByteCorpusPartWriter:
    def __init__(self, part_root: Path, *, num_shards: int, stripe_bytes: int, io_buffer_bytes: int, base_shard_index: int = 0):
        self.part_root = part_root
        self.num_shards = num_shards
        self.stripe_bytes = stripe_bytes
        self.io_buffer_bytes = io_buffer_bytes
        self.base_shard_index = base_shard_index % num_shards
        self.part_root.mkdir(parents=True, exist_ok=True)
        self.files: dict[tuple[str, int], object] = {}
        self.byte_counts = {split: [0 for _ in range(num_shards)] for split in _SPLIT_NAMES}
        self.total_bytes = {split: 0 for split in _SPLIT_NAMES}

    def close(self) -> None:
        for part_file in self.files.values():
            part_file.close()

    def _part_path(self, split: str, shard_index: int) -> Path:
        return self.part_root / f"{split}_{shard_index + 1}_of_{self.num_shards}.part"

    def _file(self, split: str, shard_index: int):
        key = (split, shard_index)
        part_file = self.files.get(key)
        if part_file is None:
            part_file = self._part_path(split, shard_index).open("wb", buffering=self.io_buffer_bytes)
            self.files[key] = part_file
        return part_file

    def _write_part(self, split: str, part) -> None:
        part_view = memoryview(part)
        offset = 0
        while offset < len(part_view):
            split_written = self.total_bytes[split]
            shard_index = (self.base_shard_index + split_written // self.stripe_bytes) % self.num_shards
            stripe_remaining = self.stripe_bytes - (split_written % self.stripe_bytes)
            chunk_size = min(len(part_view) - offset, stripe_remaining)
            part_file = self._file(split, shard_index)
            part_file.write(part_view[offset : offset + chunk_size])
            self.byte_counts[split][shard_index] += chunk_size
            self.total_bytes[split] += chunk_size
            offset += chunk_size

    def add_parts(self, split: str, parts: Sequence[object]) -> None:
        for part in parts:
            if len(part) != 0:
                self._write_part(split, part)

    def part_paths(self) -> list[tuple[str, int, str, int]]:
        paths: list[tuple[str, int, str, int]] = []
        for split in _SPLIT_NAMES:
            for shard_index in range(self.num_shards):
                byte_count = self.byte_counts[split][shard_index]
                if byte_count == 0:
                    continue
                paths.append((split, shard_index, str(self._part_path(split, shard_index)), byte_count))
        return paths


def _split_caps_for_capped_run() -> dict[str, int]:
    caps: dict[str, int] = {}
    default_targets = {
        "train": max(BYTE_LM_CONTEXT_LENGTH + 1, int(BYTE_LM_TARGET_TEXT_BYTES * 0.98)),
        "validate": max(BYTE_LM_CONTEXT_LENGTH + 1, int(BYTE_LM_TARGET_TEXT_BYTES * 0.01)),
        "test": max(BYTE_LM_CONTEXT_LENGTH + 1, BYTE_LM_TARGET_TEXT_BYTES - int(BYTE_LM_TARGET_TEXT_BYTES * 0.99)),
    }
    limits = {
        "train": BYTE_LM_TRAIN_EXAMPLES,
        "validate": BYTE_LM_VALIDATE_EXAMPLES,
        "test": BYTE_LM_TEST_EXAMPLES,
    }
    for split in _SPLIT_NAMES:
        example_limit = limits[split]
        if example_limit > 0:
            caps[split] = BYTE_LM_CONTEXT_LENGTH + 1 + (example_limit - 1) * BYTE_LM_STRIDE
        else:
            caps[split] = default_targets[split]
    return caps


def _iter_arrow_utf8_payload_parts(array):
    pa = pytest.importorskip("pyarrow", reason="FineWeb-Edu byte LM setup requires pyarrow; pip install pyarrow")
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()

    if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
        for text in array.to_pylist():
            if text is None:
                continue
            yield (memoryview(text.encode("utf-8", errors="ignore")), _DOC_SEPARATOR)
        return

    buffers = array.buffers()
    offsets_buffer = buffers[1]
    data_buffer = buffers[2]
    if offsets_buffer is None or data_buffer is None:
        return

    offset_dtype = np.int64 if pa.types.is_large_string(array.type) else np.int32
    offsets = np.frombuffer(offsets_buffer, dtype=offset_dtype)
    data = memoryview(data_buffer)
    base = array.offset
    valid = None
    if array.null_count != 0:
        valid = array.is_valid().to_numpy(zero_copy_only=False)

    for i in range(len(array)):
        if valid is not None and not bool(valid[i]):
            continue
        start = int(offsets[base + i])
        end = int(offsets[base + i + 1])
        yield (data[start:end], _DOC_SEPARATOR)


def _for_each_fineweb_payload(parquet_paths: list[Path], callback):
    pq = pytest.importorskip("pyarrow.parquet", reason="FineWeb-Edu byte LM setup requires pyarrow; pip install pyarrow")
    documents_seen = 0
    text_bytes_seen = 0
    for parquet_file_index, parquet_path in enumerate(parquet_paths, start=1):
        print(
            f"INFO byte_lm_cache: processing parquet {parquet_file_index}/{len(parquet_paths)} {parquet_path}",
            file=sys.stderr,
            flush=True,
        )
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=8192):
            for parts in _iter_arrow_utf8_payload_parts(batch.column(0)):
                size = _payload_size(parts)
                documents_seen += 1
                text_bytes_seen += size
                keep_going = callback(parts, size)
                if keep_going is False:
                    return documents_seen, text_bytes_seen
    return documents_seen, text_bytes_seen


def _process_fineweb_parquet_part_worker(args: tuple[int, int, str, str, int, int, int]) -> dict:
    parquet_file_index, parquet_file_count, parquet_path_str, part_root_str, num_shards, stripe_bytes, io_buffer_bytes = args
    pq = __import__("pyarrow.parquet", fromlist=["ParquetFile"])
    parquet_path = Path(parquet_path_str)
    part_root = Path(part_root_str) / f"parquet_{parquet_file_index:06d}"
    writer = _ByteCorpusPartWriter(
        part_root,
        num_shards=num_shards,
        stripe_bytes=stripe_bytes,
        io_buffer_bytes=io_buffer_bytes,
        base_shard_index=(parquet_file_index - 1) % num_shards,
    )
    documents_seen = 0
    text_bytes_seen = 0
    try:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=8192):
            for parts in _iter_arrow_utf8_payload_parts(batch.column(0)):
                size = _payload_size(parts)
                documents_seen += 1
                text_bytes_seen += size
                writer.add_parts(_hash_split(parts), parts)
    finally:
        writer.close()

    return {
        "index": parquet_file_index,
        "count": parquet_file_count,
        "parquet_path": str(parquet_path),
        "documents_seen": documents_seen,
        "text_bytes_seen": text_bytes_seen,
        "total_bytes": writer.total_bytes,
        "byte_counts": writer.byte_counts,
        "part_paths": writer.part_paths(),
    }


def _write_fineweb_edu_byte_corpus_shards_parallel(cache_root: Path, parquet_paths: list[Path]) -> dict:
    if not parquet_paths:
        raise RuntimeError("FineWeb-Edu byte-corpus setup found no parquet files")

    workers = min(BYTE_LM_PARQUET_WORKERS, len(parquet_paths))
    part_root = cache_root / "byte_corpus_shards_v1_uint8_shifted_labels.parts"
    if part_root.exists():
        shutil.rmtree(part_root)
    part_root.mkdir(parents=True, exist_ok=True)

    print(
        f"INFO byte_lm_cache: processing {len(parquet_paths)} parquet files with {workers} worker(s)",
        file=sys.stderr,
        flush=True,
    )
    worker_args = [
        (
            parquet_file_index,
            len(parquet_paths),
            str(parquet_path),
            str(part_root),
            BYTE_LM_NUM_SHARDS,
            BYTE_LM_SHARD_STRIPE_BYTES,
            BYTE_LM_CACHE_IO_BUFFER_BYTES,
        )
        for parquet_file_index, parquet_path in enumerate(parquet_paths, start=1)
    ]

    results: dict[int, dict] = {}
    documents_seen = 0
    text_bytes_seen = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_process_fineweb_parquet_part_worker, args): args[0]
                for args in worker_args
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                result = future.result()
                completed += 1
                index = int(result["index"])
                results[index] = result
                documents_seen += int(result["documents_seen"])
                text_bytes_seen += int(result["text_bytes_seen"])
                print(
                    "INFO byte_lm_cache: finished parquet "
                    f"{completed}/{len(parquet_paths)} index={index} "
                    f"bytes={result['text_bytes_seen']} path={result['parquet_path']}",
                    file=sys.stderr,
                    flush=True,
                )

        writer = _ByteCorpusAppendWriter(
            cache_root,
            num_shards=BYTE_LM_NUM_SHARDS,
            context_length=BYTE_LM_CONTEXT_LENGTH,
            stride=BYTE_LM_STRIDE,
            stripe_bytes=BYTE_LM_SHARD_STRIPE_BYTES,
        )
        try:
            print("INFO byte_lm_cache: concatenating parquet worker parts into final byte-corpus shards", file=sys.stderr, flush=True)
            for parquet_file_index in range(1, len(parquet_paths) + 1):
                result = results[parquet_file_index]
                for split, shard_index, part_path, byte_count in result["part_paths"]:
                    if int(byte_count) != Path(part_path).stat().st_size:
                        raise RuntimeError(f"byte-corpus part size changed while concatenating: {part_path}")
                    writer.add_shard_part(split, int(shard_index), Path(part_path))

            if any(
                _byte_corpus_num_examples(writer.total_bytes[split], context_length=BYTE_LM_CONTEXT_LENGTH, stride=BYTE_LM_STRIDE) <= 0
                for split in _SPLIT_NAMES
            ):
                raise RuntimeError(f"FineWeb-Edu byte-corpus setup produced empty split(s): {writer.total_bytes}")
            shard_paths = writer.finalize()
        finally:
            writer.close()
    finally:
        shutil.rmtree(part_root, ignore_errors=True)

    split_examples = {
        split: sum(
            _byte_corpus_num_examples(writer.byte_counts[split][shard_index], context_length=BYTE_LM_CONTEXT_LENGTH, stride=BYTE_LM_STRIDE)
            for shard_index in range(writer.num_shards)
        )
        for split in _SPLIT_NAMES
    }
    print(
        "INFO byte_lm_cache: wrote byte-corpus bytes "
        f"train={writer.total_bytes['train']} validate={writer.total_bytes['validate']} test={writer.total_bytes['test']}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "shard_paths": shard_paths,
        "train_examples": split_examples["train"],
        "validate_examples": split_examples["validate"],
        "test_examples": split_examples["test"],
        "stream_bytes": {"all": text_bytes_seen, **{split: writer.total_bytes[split] for split in _SPLIT_NAMES}},
        "parquet_files": [str(path) for path in parquet_paths],
        "documents_seen": documents_seen,
        "compact_shard_format": 1,
        "byte_corpus": True,
        "shard_stripe_bytes": BYTE_LM_SHARD_STRIPE_BYTES,
        "append_only_byte_corpus": True,
        "parallel_parquet_processing": True,
        "parquet_workers": workers,
    }


def _write_fineweb_edu_byte_corpus_shards(cache_root: Path) -> dict:
    downloads_root = cache_root / "downloads" / "fineweb_edu_sample_100bt"
    parquet_paths = _fineweb_edu_parquet_paths(downloads_root)

    if BYTE_LM_FULL_DATASET and len(parquet_paths) > 1:
        return _write_fineweb_edu_byte_corpus_shards_parallel(cache_root, parquet_paths)

    print("INFO byte_lm_cache: writing append-only byte-corpus shards", file=sys.stderr, flush=True)
    writer = _ByteCorpusAppendWriter(
        cache_root,
        num_shards=BYTE_LM_NUM_SHARDS,
        context_length=BYTE_LM_CONTEXT_LENGTH,
        stride=BYTE_LM_STRIDE,
        stripe_bytes=BYTE_LM_SHARD_STRIPE_BYTES,
    )
    capped_split_caps = _split_caps_for_capped_run()
    capped_written_bytes = {split: 0 for split in _SPLIT_NAMES}
    capped_split_index = 0

    def append_capped_payload(parts: Sequence[object], payload_size: int):
        nonlocal capped_split_index
        payload_offset = 0
        while payload_offset < payload_size and capped_split_index < len(_SPLIT_NAMES):
            split = _SPLIT_NAMES[capped_split_index]
            remaining = capped_split_caps[split] - capped_written_bytes[split]
            if remaining <= 0:
                capped_split_index += 1
                continue
            chunk_size = min(payload_size - payload_offset, remaining)
            writer.add_slice(split, parts, payload_offset, chunk_size)
            capped_written_bytes[split] += chunk_size
            payload_offset += chunk_size
            if capped_written_bytes[split] >= capped_split_caps[split]:
                capped_split_index += 1

    try:
        def callback(parts: Sequence[object], payload_size: int):
            nonlocal capped_split_index
            if BYTE_LM_FULL_DATASET:
                writer.add_parts(_hash_split(parts), parts)
                return True
            append_capped_payload(parts, payload_size)
            return capped_split_index < len(_SPLIT_NAMES)

        documents_seen, text_bytes_seen = _for_each_fineweb_payload(parquet_paths, callback)
        if not BYTE_LM_FULL_DATASET and any(capped_written_bytes[name] < capped_split_caps[name] for name in _SPLIT_NAMES):
            raise RuntimeError(
                "FineWeb-Edu byte-corpus setup did not reach requested bytes; "
                f"written={capped_written_bytes}, targets={capped_split_caps}, max_parquet_files={BYTE_LM_MAX_PARQUET_FILES}"
            )
        if any(_byte_corpus_num_examples(writer.total_bytes[split], context_length=BYTE_LM_CONTEXT_LENGTH, stride=BYTE_LM_STRIDE) <= 0 for split in _SPLIT_NAMES):
            raise RuntimeError(f"FineWeb-Edu byte-corpus setup produced empty split(s): {writer.total_bytes}")

        shard_paths = writer.finalize()
    finally:
        writer.close()

    split_examples = {
        split: sum(
            _byte_corpus_num_examples(writer.byte_counts[split][shard_index], context_length=BYTE_LM_CONTEXT_LENGTH, stride=BYTE_LM_STRIDE)
            for shard_index in range(writer.num_shards)
        )
        for split in _SPLIT_NAMES
    }
    print(
        "INFO byte_lm_cache: wrote byte-corpus bytes "
        f"train={writer.total_bytes['train']} validate={writer.total_bytes['validate']} test={writer.total_bytes['test']}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "shard_paths": shard_paths,
        "train_examples": split_examples["train"],
        "validate_examples": split_examples["validate"],
        "test_examples": split_examples["test"],
        "stream_bytes": {"all": text_bytes_seen, **{split: writer.total_bytes[split] for split in _SPLIT_NAMES}},
        "parquet_files": [str(path) for path in parquet_paths],
        "documents_seen": documents_seen,
        "compact_shard_format": 1,
        "byte_corpus": True,
        "shard_stripe_bytes": BYTE_LM_SHARD_STRIPE_BYTES,
        "append_only_byte_corpus": True,
    }


def _write_byte_corpus_shard_header(
    stream,
    *,
    record_size: int,
    train_count: int,
    validate_count: int,
    test_count: int,
    metadata_offset: int,
    metadata_bytes: int,
):
    header = struct.pack(
        "<16sIIQIIQQQQQQ",
        _SHARD_MAGIC,
        _SHARD_FORMAT_VERSION,
        _SHARD_HEADER_BYTES,
        record_size,
        _SHARD_UINT8_DTYPE,
        _SHARD_LAYOUT_BYTE_CORPUS,
        train_count,
        validate_count,
        test_count,
        1,
        metadata_offset,
        metadata_bytes,
    )
    if len(header) != _SHARD_HEADER_BYTES:
        raise RuntimeError(f"byte-corpus shard header is {len(header)} bytes; expected {_SHARD_HEADER_BYTES}")
    stream.seek(0)
    stream.write(header)



def _byte_lm_base_manifest(*, shard_paths: list[str], train_examples: int, validate_examples: int, test_examples: int, stream_info: dict) -> dict:
    return {
        "version": BYTE_LM_MANIFEST_VERSION,
        "dataset": BYTE_LM_DATASET_NAME,
        "dataset_base_url": BYTE_LM_DATASET_BASE_URL,
        "target_text_bytes": BYTE_LM_TARGET_TEXT_BYTES,
        "context_length": BYTE_LM_CONTEXT_LENGTH,
        "stride": BYTE_LM_STRIDE,
        "vocab_size": BYTE_LM_VOCAB_SIZE,
        "record_shape": [BYTE_LM_CONTEXT_LENGTH + 1],
        "example_shape": [BYTE_LM_CONTEXT_LENGTH],
        "label_shape": [BYTE_LM_CONTEXT_LENGTH],
        "example_data_type": "uint8",
        "label_data_type": "uint8",
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": test_examples,
        "num_shards": BYTE_LM_NUM_SHARDS,
        "shard_paths": shard_paths,
        **stream_info,
    }


def _ensure_byte_lm_shards() -> dict:
    manifest_path = _byte_lm_manifest_path(BYTE_LM_CACHE_DIR)
    if BYTE_LM_REBUILD and BYTE_LM_CACHE_DIR.exists():
        shutil.rmtree(BYTE_LM_CACHE_DIR)
    if manifest_path.exists() and not BYTE_LM_REBUILD:
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "version": BYTE_LM_MANIFEST_VERSION,
            "dataset": BYTE_LM_DATASET_NAME,
            "target_text_bytes": BYTE_LM_TARGET_TEXT_BYTES,
            "context_length": BYTE_LM_CONTEXT_LENGTH,
            "stride": BYTE_LM_STRIDE,
            "vocab_size": BYTE_LM_VOCAB_SIZE,
            "num_shards": BYTE_LM_NUM_SHARDS,
            "record_shape": [BYTE_LM_CONTEXT_LENGTH + 1],
            "example_shape": [BYTE_LM_CONTEXT_LENGTH],
            "label_shape": [BYTE_LM_CONTEXT_LENGTH],
            "compact_shard_format": 1,
            "byte_corpus": True,
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            if all(Path(path).exists() for path in manifest["shard_paths"]):
                return manifest

    BYTE_LM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    compact_info = _write_fineweb_edu_byte_corpus_shards(BYTE_LM_CACHE_DIR)
    shard_paths = sorted(str(Path(path)) for path in compact_info["shard_paths"])
    for path in shard_paths:
        assert Path(path).exists(), f"expected shard file {path} to exist"

    manifest = _byte_lm_base_manifest(
        shard_paths=shard_paths,
        train_examples=compact_info["train_examples"],
        validate_examples=compact_info["validate_examples"],
        test_examples=compact_info["test_examples"],
        stream_info={key: value for key, value in compact_info.items() if key not in {"shard_paths", "train_examples", "validate_examples", "test_examples"}},
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _byte_lm_loader(*, batch_size: int):
    manifest = _ensure_byte_lm_shards()
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.uint8,
        manifest["label_shape"],
        thor.DataType.uint8,
        batch_size=batch_size,
        dataset_name="fineweb_edu_byte_lm_uint8",
        batch_queue_depth=BYTE_LM_LOADER_QUEUE_DEPTH,
    )
    return loader, manifest


def _build_byte_transformer_lm(name: str):
    assert BYTE_LM_HIDDEN_DIM % BYTE_LM_NUM_HEADS == 0
    network = thor.Network(name)
    tokens_input = thor.layers.NetworkInput(network, "examples", [BYTE_LM_CONTEXT_LENGTH], thor.DataType.uint8)
    labels_input = thor.layers.NetworkInput(network, "labels", [BYTE_LM_CONTEXT_LENGTH], thor.DataType.uint8)
    tokens = tokens_input.get_feature_output()
    labels = labels_input.get_feature_output()

    x = thor.layers.Embedding(
        network,
        tokens,
        vocabulary_size=BYTE_LM_VOCAB_SIZE,
        embedding_dim=BYTE_LM_HIDDEN_DIM,
        weights_data_type=thor.DataType.fp16,
        sparse_gradients=True,
    ).get_feature_output()

    head_dim = BYTE_LM_HIDDEN_DIM // BYTE_LM_NUM_HEADS
    rotary_dim = head_dim if head_dim % 2 == 0 else head_dim - 1
    for layer_index in range(BYTE_LM_NUM_LAYERS):
        norm = thor.layers.RMSNorm(network, x, parameter_data_type=thor.DataType.fp32)
        attention = thor.layers.Attention(
            network,
            norm.get_feature_output(),
            num_heads=BYTE_LM_NUM_HEADS,
            head_dim=head_dim,
            output_features=BYTE_LM_HIDDEN_DIM,
            has_bias=False,
            mask_kind="causal_top_left",
            use_rope=rotary_dim > 0,
            rope_rotary_dim=rotary_dim,
            weights_data_type=thor.DataType.fp16,
            compute_data_type=thor.DataType.fp32,
            output_data_type=thor.DataType.fp16,
            rope_in_place=True,
        )
        x = _ResidualAdd(network, x, attention.get_feature_output())["feature_output"]

    norm = thor.layers.RMSNorm(network, x, parameter_data_type=thor.DataType.fp32)
    logits = thor.layers.FullyConnected(
        network,
        norm.get_feature_output(),
        BYTE_LM_VOCAB_SIZE,
        has_bias=True,
        activation=None,
        preserve_prefix_dimensions=True,
    ).get_feature_output()
    assert logits.get_dimensions() == [BYTE_LM_CONTEXT_LENGTH, BYTE_LM_VOCAB_SIZE]
    # Sparse CE is tokenwise for [B, S, V] logits / [B, S] labels.
    # Thor optimizers normalize gradients by the API batch dimension, so scale
    # this sequence loss by 1/S to get the usual mean-per-token LM objective.
    loss = thor.losses.SparseCategoricalCrossEntropy(
        network,
        logits,
        labels,
        BYTE_LM_VOCAB_SIZE,
        thor.DataType.fp32,
        loss_weight=BYTE_LM_LOSS_WEIGHT,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "next_byte_logits", logits, thor.DataType.fp16)
    return network


def test_queued_trainer_trains_byte_level_transformer_lm_on_fineweb_edu(capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _byte_lm_loader(batch_size=BYTE_LM_BATCH_SIZE)
        assert loader.get_num_train_examples() > 0
        assert loader.get_num_validate_examples() > 0
        assert loader.get_num_train_batches() > 0
        assert loader.get_num_validate_batches() > 0
        assert manifest["record_shape"] == [BYTE_LM_CONTEXT_LENGTH + 1]
        assert manifest["example_shape"] == [BYTE_LM_CONTEXT_LENGTH]
        assert manifest["label_shape"] == [BYTE_LM_CONTEXT_LENGTH]

        network = _build_byte_transformer_lm("python_integration_fineweb_edu_byte_transformer_lm")
        optimizer = thor.optimizers.AdamW(
            alpha=BYTE_LM_LEARNING_RATE,
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-8,
            weight_decay=BYTE_LM_WEIGHT_DECAY,
        )
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats=True,
            stats_interval_s=BYTE_LM_STATS_INTERVAL_S,
            max_in_flight_batches=BYTE_LM_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color=BYTE_LM_STATS_COLOR,
            save_model_dir=str(BYTE_LM_SAVE_DIR) if BYTE_LM_SAVE_DIR is not None else None,
            save_model_overwrite=BYTE_LM_SAVE_OVERWRITE,
            save_optimizer_state=BYTE_LM_SAVE_OPTIMIZER_STATE,
        )
        stats = _fit_and_capture_stats(trainer, epochs=BYTE_LM_EPOCHS)
        _assert_finite_positive_losses(stats, model_name="fineweb_edu_byte_transformer_lm")
        assert max(_flops_value(entry["flops_per_s"]) for entry in stats) > 0.0
        if BYTE_LM_SAVE_DIR is not None:
            assert (BYTE_LM_SAVE_DIR / f"{network.get_network_name()}.thor.tar").exists()
