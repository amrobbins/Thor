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
from pathlib import Path

import numpy as np
import pytest
import thor

RUN_UCF101_3D_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_UCF101_3D_INTEGRATION") == "1"
UCF101_3D_CACHE_DIR = Path(os.environ.get("THOR_UCF101_3D_CACHE_DIR", "/tmp/thor_ucf101_3d_training"))
UCF101_3D_DATASET_URL = os.environ.get(
    "THOR_UCF101_3D_DATASET_URL",
    "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz",
)
UCF101_3D_CLIP_FRAMES = int(os.environ.get("THOR_UCF101_3D_CLIP_FRAMES", "16"))
UCF101_3D_IMAGE_SIZE = int(os.environ.get("THOR_UCF101_3D_IMAGE_SIZE", "112"))
UCF101_3D_BATCH_SIZE = int(os.environ.get("THOR_UCF101_3D_BATCH_SIZE", "16"))
UCF101_3D_EPOCHS = int(os.environ.get("THOR_UCF101_3D_EPOCHS", "10"))
UCF101_3D_MAX_IN_FLIGHT_BATCHES = int(os.environ.get("THOR_UCF101_3D_MAX_IN_FLIGHT_BATCHES", "8"))
UCF101_3D_STATS_INTERVAL_S = float(os.environ.get("THOR_UCF101_3D_STATS_INTERVAL_S", "0.0"))
UCF101_3D_REBUILD = os.environ.get("THOR_UCF101_3D_REBUILD") == "1"
UCF101_3D_NUM_SHARDS = int(os.environ.get("THOR_UCF101_3D_NUM_SHARDS", "1"))
# Default 0 means use the full subset.  Non-zero values are useful while
# iterating on the test itself without changing the network under test.
UCF101_3D_MAX_VIDEOS_PER_CLASS = int(os.environ.get("THOR_UCF101_3D_MAX_VIDEOS_PER_CLASS", "0"))
# These tests are intentionally gated and should prove useful training, not just
# graph execution.  Compare moving windows so assertions do not depend on any
# single randomly sampled batch.
UCF101_3D_TRAINING_ASSERTION_WINDOW = int(os.environ.get("THOR_UCF101_3D_TRAINING_ASSERTION_WINDOW", "4"))
UCF101_3D_MAX_FINAL_TRAIN_LOSS_RATIO = float(os.environ.get("THOR_UCF101_3D_MAX_FINAL_TRAIN_LOSS_RATIO", "0.80"))
UCF101_3D_LEARNING_RATE = float(os.environ.get("THOR_UCF101_3D_LEARNING_RATE", "0.01"))
UCF101_3D_MOMENTUM = float(os.environ.get("THOR_UCF101_3D_MOMENTUM", "0.9"))
UCF101_3D_STATS_COLOR = os.environ.get("THOR_UCF101_3D_STATS_COLOR", "always").lower()
assert UCF101_3D_STATS_COLOR in {"always", "auto", "never"}
UCF101_3D_MANIFEST_VERSION = 1

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.ucf101_3d_integration,
    pytest.mark.skipif(
        not RUN_UCF101_3D_INTEGRATION,
        reason="set THOR_RUN_TRAINING_UCF101_3D_INTEGRATION=1 to run heavyweight UCF101 3D video training tests",
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+)")


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
                prefix=f"thor_ucf101_3d_fit_fd{fd}_", suffix=".log", delete=False)
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


def _fit_and_capture_stats(trainer, *, epochs: int):
    with _NativeOutputTee() as tee:
        trainer.fit(epochs=epochs)
    captured_text = tee.text()
    stats = _captured_trainer_stats(captured_text)

    assert stats, "trainer emitted no parseable stats; set THOR_UCF101_3D_STATS_INTERVAL_S=0.0 if this happens"
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
    assert any(
        entry["phase"] == "train"
        for entry in stats), f"{model_name}: no train stats reported; phase_counts={phase_counts}"
    assert any(
        entry["phase"] == "validate"
        for entry in stats), f"{model_name}: no validate stats reported; phase_counts={phase_counts}"


def _mean(values):
    return sum(values) / len(values)


def _assert_train_loss_improves(stats, *, model_name: str):
    train_losses = [entry["loss"] for entry in stats if entry["phase"] == "train"]
    phase_counts = _stats_phase_counts(stats)
    window = max(1, UCF101_3D_TRAINING_ASSERTION_WINDOW)
    assert len(train_losses) >= 2 * window, (
        f"{model_name}: not enough train loss reports to assert useful training; "
        f"got={len(train_losses)} required={2 * window} phase_counts={phase_counts}; "
        "increase epochs or lower THOR_UCF101_3D_STATS_INTERVAL_S")

    initial = _mean(train_losses[:window])
    final = _mean(train_losses[-window:])
    max_allowed_final = initial * UCF101_3D_MAX_FINAL_TRAIN_LOSS_RATIO
    assert final <= max_allowed_final, (
        f"{model_name}: train loss did not improve enough; "
        f"initial_window_mean={initial:.6f} final_window_mean={final:.6f} "
        f"required_final<={max_allowed_final:.6f} "
        f"max_final_ratio={UCF101_3D_MAX_FINAL_TRAIN_LOSS_RATIO:.3f} "
        f"window={window} train_losses={train_losses}")


def _safe_extract_tar(archive: tarfile.TarFile, dest_dir: Path):
    dest_resolved = dest_dir.resolve()
    for member in archive.getmembers():
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise RuntimeError(f"refusing to extract path outside cache directory: {member.name}")
    archive.extractall(dest_resolved)


def _ensure_ucf101_subset_downloaded(cache_root: Path) -> Path:
    dataset_root = cache_root / "UCF101_subset"
    if dataset_root.exists() and not UCF101_3D_REBUILD:
        return dataset_root

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / "UCF101_subset.tar.gz"
    if UCF101_3D_REBUILD and dataset_root.exists():
        shutil.rmtree(dataset_root)
    if not archive_path.exists() or UCF101_3D_REBUILD:
        urllib.request.urlretrieve(UCF101_3D_DATASET_URL, archive_path)

    try:
        with tarfile.open(archive_path, "r:*") as archive:
            _safe_extract_tar(archive, cache_root)
    except tarfile.TarError as exc:
        magic = archive_path.read_bytes()[:32]
        raise RuntimeError(
            f"UCF101 subset archive is not a readable tar archive: {archive_path}; first bytes={magic!r}") from exc

    if not dataset_root.exists():
        candidates = [path for path in cache_root.iterdir() if path.is_dir() and (path / "train").exists()]
        if len(candidates) == 1:
            dataset_root = candidates[0]
        else:
            raise RuntimeError(f"UCF101 subset extraction did not create expected directory {dataset_root}")
    return dataset_root


def _video_paths_for_split(dataset_root: Path, split_name: str):
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        raise RuntimeError(f"UCF101 subset split directory does not exist: {split_dir}")
    paths = []
    for suffix in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
        paths.extend(split_dir.glob(f"*/{suffix}"))
    paths = sorted(path for path in paths if path.is_file())
    if not paths:
        raise RuntimeError(f"UCF101 subset split {split_name} has no video files under {split_dir}")
    return paths


def _ucf101_class_names(dataset_root: Path) -> list[str]:
    class_names = set()
    for split_name in ("train", "val", "test"):
        for video_path in _video_paths_for_split(dataset_root, split_name):
            class_names.add(video_path.parent.name)
    if not class_names:
        raise RuntimeError(f"UCF101 subset has no class directories under {dataset_root}")
    return sorted(class_names)


def _decode_video_rgb_frames_with_cv2(video_path: Path):
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            else:
                frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        capture.release()
    return frames if frames else None


def _decode_video_rgb_frames_with_imageio(video_path: Path):
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError:
        return None

    frames = []
    try:
        for frame in iio.imiter(video_path):
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.shape[-1] > 3:
                frame = frame[:, :, :3]
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
    except Exception:
        return None
    return frames if frames else None


def _ffprobe_video_size(video_path: Path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        width_text, height_text = first_line.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, IndexError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _decode_video_rgb_frames_with_ffmpeg(video_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    size = _ffprobe_video_size(video_path)
    if size is None:
        return None
    width, height = size
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    frame_bytes = width * height * 3
    if frame_bytes <= 0 or len(result.stdout) < frame_bytes:
        return None
    usable_bytes = len(result.stdout) - (len(result.stdout) % frame_bytes)
    if usable_bytes == 0:
        return None
    frames = np.frombuffer(result.stdout[:usable_bytes], dtype=np.uint8).reshape((-1, height, width, 3))
    return [np.ascontiguousarray(frame) for frame in frames]


def _decode_video_rgb_frames(video_path: Path):
    frames = _decode_video_rgb_frames_with_cv2(video_path)
    if frames is None:
        frames = _decode_video_rgb_frames_with_imageio(video_path)
    if frames is None:
        frames = _decode_video_rgb_frames_with_ffmpeg(video_path)
    if frames is None:
        raise RuntimeError(
            "Unable to decode UCF101 video. Install opencv-python or imageio[ffmpeg], or make sure the ffmpeg "
            f"and ffprobe executables are on PATH. Video: {video_path}")
    return frames


def _uniform_frame_indices(num_frames: int, num_samples: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("cannot sample an empty video")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_frames == 1:
        return np.zeros((num_samples,), dtype=np.int64)
    return np.rint(np.linspace(0, num_frames - 1, num_samples)).astype(np.int64)


def _video_to_cthw_fp16(video_path: Path, *, clip_frames: int, image_size: int) -> np.ndarray:
    from PIL import Image  # type: ignore

    frames = _decode_video_rgb_frames(video_path)
    selected = _uniform_frame_indices(len(frames), clip_frames)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    processed_frames = []
    for index in selected:
        frame = np.asarray(frames[int(index)], dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]
        image = Image.fromarray(frame, mode="RGB").resize((image_size, image_size), resampling)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        processed_frames.append(arr)

    # Thor Convolution3d expects per-example tensor dimensions [C, D, H, W]
    # without batch; D is the temporal clip dimension.
    t_h_w_c = np.stack(processed_frames, axis=0)
    c_t_h_w = np.transpose(t_h_w_c, (3, 0, 1, 2))
    return np.ascontiguousarray(c_t_h_w, dtype=np.float16)


def _ucf101_manifest_path(cache_root: Path) -> Path:
    return cache_root / f"ucf101_subset_{UCF101_3D_CLIP_FRAMES}x{UCF101_3D_IMAGE_SIZE}_cthw_fp16_manifest.json"


def _ucf101_shard_root(cache_root: Path) -> Path:
    return cache_root / f"ucf101_subset_shards_{UCF101_3D_CLIP_FRAMES}x{UCF101_3D_IMAGE_SIZE}_cthw_fp16"


def _ucf101_base_manifest(*, shard_paths, class_names, train_examples: int, validate_examples: int, test_examples: int):
    return {
        "version": UCF101_3D_MANIFEST_VERSION,
        "dataset_url": UCF101_3D_DATASET_URL,
        "clip_frames": UCF101_3D_CLIP_FRAMES,
        "image_size": UCF101_3D_IMAGE_SIZE,
        "dtype": "fp16",
        "max_videos_per_class": UCF101_3D_MAX_VIDEOS_PER_CLASS,
        "num_classes": len(class_names),
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": test_examples,
        "example_shape": [3, UCF101_3D_CLIP_FRAMES, UCF101_3D_IMAGE_SIZE, UCF101_3D_IMAGE_SIZE],
        "label_shape": [len(class_names)],
        "shard_paths": sorted(str(Path(path)) for path in shard_paths),
        "label_names": list(class_names),
    }


def _read_ucf101_manifest_if_valid(cache_root: Path):
    manifest_file = _ucf101_manifest_path(cache_root)
    if UCF101_3D_REBUILD or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None
    expected = {
        "version": UCF101_3D_MANIFEST_VERSION,
        "dataset_url": UCF101_3D_DATASET_URL,
        "clip_frames": UCF101_3D_CLIP_FRAMES,
        "image_size": UCF101_3D_IMAGE_SIZE,
        "dtype": "fp16",
        "max_videos_per_class": UCF101_3D_MAX_VIDEOS_PER_CLASS,
        "example_shape": [3, UCF101_3D_CLIP_FRAMES, UCF101_3D_IMAGE_SIZE, UCF101_3D_IMAGE_SIZE],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    if not manifest.get("shard_paths") or not manifest.get("label_names"):
        return None
    return manifest


def _write_ucf101_split(
    dataset_root: Path,
    *,
    source_split_name: str,
    dest_split_name: str,
    class_names: list[str],
    raw_root: Path,
    clip_frames: int,
    image_size: int,
) -> int:
    for class_name in class_names:
        (raw_root / dest_split_name / class_name).mkdir(parents=True, exist_ok=True)

    num_written = 0
    for class_name in class_names:
        class_dir = dataset_root / source_split_name / class_name
        video_paths = [] if not class_dir.exists() else sorted(
            path for suffix in ("*.avi", "*.mp4", "*.mov", "*.mkv") for path in class_dir.glob(suffix))
        if UCF101_3D_MAX_VIDEOS_PER_CLASS > 0:
            video_paths = video_paths[:UCF101_3D_MAX_VIDEOS_PER_CLASS]
        for video_path in video_paths:
            clip = _video_to_cthw_fp16(video_path, clip_frames=clip_frames, image_size=image_size)
            filename = raw_root / dest_split_name / class_name / f"{dest_split_name}_{video_path.stem}.bin"
            filename.write_bytes(clip.tobytes(order="C"))
            num_written += 1

    if num_written == 0:
        raise RuntimeError(f"UCF101 split {source_split_name} produced no usable video examples")
    return num_written


def _ensure_ucf101_3d_shards():
    UCF101_3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_ucf101_manifest_if_valid(UCF101_3D_CACHE_DIR)
    if manifest is not None:
        return manifest

    try:
        import PIL  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The heavyweight UCF101 3D integration tests require Pillow. "
            "Install Pillow plus either opencv-python or imageio[ffmpeg] in the test venv.") from exc

    dataset_root = _ensure_ucf101_subset_downloaded(UCF101_3D_CACHE_DIR)
    class_names = _ucf101_class_names(dataset_root)
    processing_root = UCF101_3D_CACHE_DIR / "processing_tmp"
    raw_root = processing_root / "raw_cthw_fp16"
    shard_root = _ucf101_shard_root(UCF101_3D_CACHE_DIR)
    base_name = f"ucf101_subset_{UCF101_3D_CLIP_FRAMES}x{UCF101_3D_IMAGE_SIZE}_cthw_fp16"

    if processing_root.exists():
        shutil.rmtree(processing_root)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_dest_dirs = []
    for shard_index in range(UCF101_3D_NUM_SHARDS):
        dest = shard_root / f"dest_{shard_index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs.append(dest)

    train_count = _write_ucf101_split(
        dataset_root,
        source_split_name="train",
        dest_split_name="train",
        class_names=class_names,
        raw_root=raw_root,
        clip_frames=UCF101_3D_CLIP_FRAMES,
        image_size=UCF101_3D_IMAGE_SIZE,
    )
    validate_count = _write_ucf101_split(
        dataset_root,
        source_split_name="val",
        dest_split_name="validate",
        class_names=class_names,
        raw_root=raw_root,
        clip_frames=UCF101_3D_CLIP_FRAMES,
        image_size=UCF101_3D_IMAGE_SIZE,
    )
    test_count = _write_ucf101_split(
        dataset_root,
        source_split_name="test",
        dest_split_name="test",
        class_names=class_names,
        raw_root=raw_root,
        clip_frames=UCF101_3D_CLIP_FRAMES,
        image_size=UCF101_3D_IMAGE_SIZE,
    )

    example_size_in_bytes = (
        3 * UCF101_3D_CLIP_FRAMES * UCF101_3D_IMAGE_SIZE * UCF101_3D_IMAGE_SIZE * np.dtype(np.float16).itemsize)
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

    manifest = _ucf101_base_manifest(
        shard_paths=shard_paths,
        class_names=class_names,
        train_examples=train_count,
        validate_examples=validate_count,
        test_examples=test_count,
    )
    _ucf101_manifest_path(UCF101_3D_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(processing_root)
    return manifest


def _ucf101_3d_loader(*, batch_size: int):
    manifest = _ensure_ucf101_3d_shards()
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.fp16,
        manifest["label_shape"],
        thor.DataType.fp16,
        batch_size=batch_size,
        dataset_name="ucf101_subset_cthw_fp16_video",
    )
    return loader, manifest


def _serializable_relu(network: thor.Network, x: thor.Tensor) -> thor.Tensor:
    return thor.activations.Relu().add_to_network(network, x)


def _conv3d_bn_relu(
    network: thor.Network,
    x: thor.Tensor,
    out_channels: int,
    kernel_depth: int,
    kernel_height: int,
    kernel_width: int,
    *,
    depth_stride: int = 1,
    vertical_stride: int = 1,
    horizontal_stride: int = 1,
    depth_padding: int = 0,
    vertical_padding: int = 0,
    horizontal_padding: int = 0,
    dtype=thor.DataType.fp16,
) -> thor.Tensor:
    conv = thor.layers.Convolution3d(
        network,
        x,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        depth_stride=depth_stride,
        vertical_stride=vertical_stride,
        horizontal_stride=horizontal_stride,
        depth_padding=depth_padding,
        vertical_padding=vertical_padding,
        horizontal_padding=horizontal_padding,
        has_bias=False,
        activation=None,
    )
    bn = thor.layers.BatchNormalization(network, conv.get_feature_output())
    return _serializable_relu(network, bn.get_feature_output())


def _build_c3d_ucf101(name: str, *, num_classes: int, clip_frames: int, image_size: int, dtype=thor.DataType.fp16):
    assert clip_frames >= 16, "C3D integration test expects temporal depth >= 16"
    assert image_size >= 112, "C3D integration test expects spatial size >= 112"
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [3, clip_frames, image_size, image_size], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)

    x = examples.get_feature_output()
    # C3D-style all-3D stem.  Because Thor only exposes 2D Pooling today, the
    # integration test uses strided 3D convolutions for temporal and spatial
    # downsampling while keeping real temporal extent throughout the early blocks.
    x = _conv3d_bn_relu(
        network,
        x,
        64,
        3,
        7,
        7,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=3,
        horizontal_padding=3,
        dtype=dtype)
    x = _conv3d_bn_relu(network, x, 64, 3, 3, 3, depth_padding=1, vertical_padding=1, horizontal_padding=1, dtype=dtype)
    x = _conv3d_bn_relu(
        network,
        x,
        128,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype)
    x = _conv3d_bn_relu(
        network,
        x,
        256,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype)
    x = _conv3d_bn_relu(
        network,
        x,
        512,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype)
    x = _conv3d_bn_relu(
        network,
        x,
        512,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype)

    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    fc6 = thor.layers.FullyConnected(network, x, 4096, True, activation=thor.activations.Relu())
    x = fc6.get_feature_output()
    drop6 = thor.layers.DropOut(network, x, 0.5)
    x = drop6.get_feature_output()
    fc7 = thor.layers.FullyConnected(network, x, 4096, True, activation=thor.activations.Relu())
    x = fc7.get_feature_output()
    drop7 = thor.layers.DropOut(network, x, 0.5)
    logits = thor.layers.FullyConnected(network, drop7.get_feature_output(), num_classes, True, activation=None)

    loss = thor.losses.CategoricalCrossEntropy(
        network, logits.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def _build_r2plus1d_ucf101(name: str, *, num_classes: int, clip_frames: int, image_size: int, dtype=thor.DataType.fp16):
    assert clip_frames >= 16, "R(2+1)D integration test expects temporal depth >= 16"
    assert image_size >= 112, "R(2+1)D integration test expects spatial size >= 112"
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [3, clip_frames, image_size, image_size], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)

    x = examples.get_feature_output()

    def block(x: thor.Tensor, out_channels: int, *, downsample: bool) -> thor.Tensor:
        stride = 2 if downsample else 1
        x = _conv3d_bn_relu(
            network,
            x,
            out_channels,
            1,
            3,
            3,
            vertical_stride=stride,
            horizontal_stride=stride,
            vertical_padding=1,
            horizontal_padding=1,
            dtype=dtype,
        )
        return _conv3d_bn_relu(
            network,
            x,
            out_channels,
            3,
            1,
            1,
            depth_stride=stride,
            depth_padding=1,
            dtype=dtype,
        )

    # A compact but real spatiotemporal factorized network: every stage has a
    # spatial 1x3x3 convolution followed by a temporal 3x1x1 convolution.
    x = block(x, 64, downsample=False)
    x = block(x, 128, downsample=True)
    x = block(x, 256, downsample=True)
    x = block(x, 512, downsample=True)
    x = block(x, 512, downsample=True)

    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    fc = thor.layers.FullyConnected(network, x, 2048, True, activation=thor.activations.Relu())
    drop = thor.layers.DropOut(network, fc.get_feature_output(), 0.5)
    logits = thor.layers.FullyConnected(network, drop.get_feature_output(), num_classes, True, activation=None)

    loss = thor.losses.CategoricalCrossEntropy(
        network, logits.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def _run_full_ucf101_3d_model_training(model_builder, *, model_name: str, capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _ucf101_3d_loader(batch_size=UCF101_3D_BATCH_SIZE)
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "UCF101 3D train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "UCF101 3D validation split unexpectedly has zero batches"
        assert manifest["num_classes"] == manifest["label_shape"][0]
        assert manifest["example_shape"][1] == UCF101_3D_CLIP_FRAMES

        network = model_builder(
            f"python_integration_{model_name}_ucf101_3d_full",
            num_classes=manifest["num_classes"],
            clip_frames=manifest["clip_frames"],
            image_size=manifest["image_size"],
        )
        optimizer = thor.optimizers.Sgd(initial_learning_rate=UCF101_3D_LEARNING_RATE, momentum=UCF101_3D_MOMENTUM)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats=True,
            stats_interval_s=UCF101_3D_STATS_INTERVAL_S,
            max_in_flight_batches=UCF101_3D_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color=UCF101_3D_STATS_COLOR,
        )
        stats = _fit_and_capture_stats(trainer, epochs=UCF101_3D_EPOCHS)
        _assert_finite_positive_losses(stats, model_name=model_name)
        _assert_train_loss_improves(stats, model_name=model_name)


@pytest.mark.parametrize(
    ("model_name", "model_builder"),
    [
        ("c3d", _build_c3d_ucf101),
        ("r2plus1d", _build_r2plus1d_ucf101),
    ],
)
def test_queued_trainer_trains_real_3d_video_networks_on_ucf101_end_to_end(model_name, model_builder, capfd):
    _run_full_ucf101_3d_model_training(model_builder, model_name=model_name, capfd=capfd)
