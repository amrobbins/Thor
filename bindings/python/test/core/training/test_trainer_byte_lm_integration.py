import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

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
BYTE_LM_NUM_SHARDS = _env_int("THOR_BYTE_LM_NUM_SHARDS", 1)
BYTE_LM_TRAIN_EXAMPLES = _env_int("THOR_BYTE_LM_TRAIN_EXAMPLES", 131072)
BYTE_LM_VALIDATE_EXAMPLES = _env_int("THOR_BYTE_LM_VALIDATE_EXAMPLES", 4096)
BYTE_LM_TEST_EXAMPLES = _env_int("THOR_BYTE_LM_TEST_EXAMPLES", 4096)
BYTE_LM_TARGET_TEXT_BYTES = _env_int(
    "THOR_BYTE_LM_TARGET_TEXT_BYTES",
    max(
        1,
        (BYTE_LM_TRAIN_EXAMPLES + BYTE_LM_VALIDATE_EXAMPLES + BYTE_LM_TEST_EXAMPLES) * BYTE_LM_STRIDE
        + 3 * (BYTE_LM_CONTEXT_LENGTH + 1),
    ),
)
BYTE_LM_MAX_PARQUET_FILES = _env_int("THOR_BYTE_LM_MAX_PARQUET_FILES", 64)
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
# Bump whenever the on-disk raw shard format changes so stale /tmp caches are rebuilt.
BYTE_LM_MANIFEST_VERSION = 6

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




def _byte_lm_available_windows(path: str | Path, context_length: int, stride: int) -> int:
    if context_length <= 0:
        raise ValueError("context_length must be >= 1")
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    byte_count = Path(path).stat().st_size
    if byte_count <= context_length:
        return 0
    return 1 + (byte_count - context_length - 1) // stride


def _write_byte_lm_raw_split(
    binary_path: str | Path,
    raw_root: Path,
    split_name: str,
    requested_examples: int,
    *,
    context_length: int,
    stride: int,
) -> int:
    available = _byte_lm_available_windows(binary_path, context_length, stride)
    example_count = available if requested_examples == 0 or requested_examples > available else requested_examples
    if example_count <= 0:
        raise RuntimeError(f"byte LM split {split_name} has no available context/label windows")

    split_root = raw_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    class_dir = split_root / "sequence"
    class_dir.mkdir(parents=True, exist_ok=True)
    window_size = context_length + 1
    with Path(binary_path).open("rb") as input_file:
        for index in range(example_count):
            input_file.seek(index * stride)
            window = input_file.read(window_size)
            if len(window) != window_size:
                raise RuntimeError(f"byte LM input file ended early while reading {split_name}")
            # Inline payload format consumed by LocalBatchLoader:
            #   uint8 tokens[context_length], uint8 labels[context_length]
            # Keep the shard payload byte-native.  The model/loss support uint8
            # class-index tensors directly for byte-level language modeling, so
            # widening to uint32 here would waste IO bandwidth, cache, and pinned
            # loader memory before the first GPU op even starts.
            payload = bytes(window[:-1]) + bytes(window[1:])
            expected_payload_bytes = 2 * context_length
            if len(payload) != expected_payload_bytes:
                raise RuntimeError(
                    f"byte LM raw record has {len(payload)} bytes; expected {expected_payload_bytes}"
                )
            (class_dir / f"{split_name}_{index:012d}.bin").write_bytes(payload)
    return example_count

def _write_fineweb_edu_byte_streams(cache_root: Path) -> dict:
    pq = pytest.importorskip("pyarrow.parquet", reason="FineWeb-Edu byte LM setup requires pyarrow; pip install pyarrow")

    streams_root = cache_root / "byte_streams"
    downloads_root = cache_root / "downloads" / "fineweb_edu_sample_100bt"
    streams_root.mkdir(parents=True, exist_ok=True)
    train_path = streams_root / "train.bin"
    validate_path = streams_root / "validate.bin"
    test_path = streams_root / "test.bin"

    split_targets = {
        "train": max(BYTE_LM_CONTEXT_LENGTH + 1, int(BYTE_LM_TARGET_TEXT_BYTES * 0.98)),
        "validate": max(BYTE_LM_CONTEXT_LENGTH + 1, int(BYTE_LM_TARGET_TEXT_BYTES * 0.01)),
        "test": max(BYTE_LM_CONTEXT_LENGTH + 1, BYTE_LM_TARGET_TEXT_BYTES - int(BYTE_LM_TARGET_TEXT_BYTES * 0.99)),
    }
    split_paths = {"train": train_path, "validate": validate_path, "test": test_path}
    written = {name: 0 for name in split_targets}
    split_order = ["train", "validate", "test"]
    split_index = 0

    writers = {name: path.open("wb") for name, path in split_paths.items()}
    try:
        for parquet_index in range(BYTE_LM_MAX_PARQUET_FILES):
            parquet_path = _download_parquet_if_missing(parquet_index, downloads_root)
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(columns=["text"], batch_size=2048):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    if text is None:
                        continue
                    payload = text.encode("utf-8", errors="ignore") + b"\n\n"
                    offset = 0
                    while offset < len(payload) and split_index < len(split_order):
                        split_name = split_order[split_index]
                        remaining = split_targets[split_name] - written[split_name]
                        if remaining <= 0:
                            split_index += 1
                            continue
                        chunk = payload[offset : offset + remaining]
                        writers[split_name].write(chunk)
                        written[split_name] += len(chunk)
                        offset += len(chunk)
                        if written[split_name] >= split_targets[split_name]:
                            split_index += 1
                    if split_index >= len(split_order):
                        break
                if split_index >= len(split_order):
                    break
            if split_index >= len(split_order):
                break
    finally:
        for writer in writers.values():
            writer.close()

    if any(written[name] < split_targets[name] for name in split_targets):
        raise RuntimeError(
            "FineWeb-Edu byte stream setup did not reach requested bytes; "
            f"written={written}, targets={split_targets}, max_parquet_files={BYTE_LM_MAX_PARQUET_FILES}"
        )

    return {
        "train_binary_path": str(train_path),
        "validate_binary_path": str(validate_path),
        "test_binary_path": str(test_path),
        "stream_bytes": written,
    }


def _byte_lm_base_manifest(*, shard_paths: list[str], train_examples: int, validate_examples: int, test_examples: int, stream_info: dict) -> dict:
    return {
        "version": BYTE_LM_MANIFEST_VERSION,
        "dataset": BYTE_LM_DATASET_NAME,
        "dataset_base_url": BYTE_LM_DATASET_BASE_URL,
        "target_text_bytes": BYTE_LM_TARGET_TEXT_BYTES,
        "context_length": BYTE_LM_CONTEXT_LENGTH,
        "stride": BYTE_LM_STRIDE,
        "vocab_size": BYTE_LM_VOCAB_SIZE,
        "record_shape": [2 * BYTE_LM_CONTEXT_LENGTH],
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
            "record_shape": [2 * BYTE_LM_CONTEXT_LENGTH],
            "example_shape": [BYTE_LM_CONTEXT_LENGTH],
            "label_shape": [BYTE_LM_CONTEXT_LENGTH],
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            if all(Path(path).exists() for path in manifest["shard_paths"]):
                return manifest

    BYTE_LM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stream_info = _write_fineweb_edu_byte_streams(BYTE_LM_CACHE_DIR)
    shard_root = BYTE_LM_CACHE_DIR / "shards_raw_v6_uint8_byte_lm_inline_labels"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_dest_dirs = []
    for shard_index in range(BYTE_LM_NUM_SHARDS):
        dest = shard_root / f"dest_{shard_index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs.append(dest)

    raw_root = BYTE_LM_CACHE_DIR / "raw_uint8_byte_lm"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    actual_train_examples = _write_byte_lm_raw_split(
        stream_info["train_binary_path"],
        raw_root,
        "train",
        BYTE_LM_TRAIN_EXAMPLES,
        context_length=BYTE_LM_CONTEXT_LENGTH,
        stride=BYTE_LM_STRIDE,
    )
    actual_validate_examples = _write_byte_lm_raw_split(
        stream_info["validate_binary_path"],
        raw_root,
        "validate",
        BYTE_LM_VALIDATE_EXAMPLES,
        context_length=BYTE_LM_CONTEXT_LENGTH,
        stride=BYTE_LM_STRIDE,
    )
    actual_test_examples = _write_byte_lm_raw_split(
        stream_info["test_binary_path"],
        raw_root,
        "test",
        BYTE_LM_TEST_EXAMPLES,
        context_length=BYTE_LM_CONTEXT_LENGTH,
        stride=BYTE_LM_STRIDE,
    )

    try:
        shard_paths = thor.training.create_sharded_raw_dataset(
            [str(raw_root)],
            [str(path) for path in shard_dest_dirs],
            "fineweb_edu_byte_lm_uint8",
            2 * BYTE_LM_CONTEXT_LENGTH,
            thor.DataType.uint8,
        )
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)
    shard_paths = sorted(str(Path(path)) for path in shard_paths)
    for path in shard_paths:
        assert Path(path).exists(), f"expected shard file {path} to exist"

    manifest = _byte_lm_base_manifest(
        shard_paths=shard_paths,
        train_examples=actual_train_examples,
        validate_examples=actual_validate_examples,
        test_examples=actual_test_examples,
        stream_info=stream_info,
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
        assert manifest["record_shape"] == [2 * BYTE_LM_CONTEXT_LENGTH]
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
