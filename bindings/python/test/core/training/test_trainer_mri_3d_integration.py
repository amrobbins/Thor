import csv
import ctypes
import gzip
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
import zipfile
from pathlib import Path

import numpy as np
import pytest
import thor
from integration_flags import integration_flag_enabled, integration_skip_reason

RUN_MRI_3D_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_MRI_3D_INTEGRATION")
MRI_3D_REBUILD = os.environ.get("THOR_MRI_3D_REBUILD") == "1"
MRI_3D_STATS_INTERVAL_S = float(os.environ.get("THOR_MRI_3D_STATS_INTERVAL_S", "0.0"))
MRI_3D_MAX_IN_FLIGHT_BATCHES = int(os.environ.get("THOR_MRI_3D_MAX_IN_FLIGHT_BATCHES", "2"))
MRI_3D_EPOCHS = int(os.environ.get("THOR_MRI_3D_EPOCHS", "1"))

MRNET_CACHE_DIR = Path(os.environ.get("THOR_MRNET_CACHE_DIR", "/tmp/thor_mrnet_3d_training"))
MRNET_ROOT = os.environ.get("THOR_MRNET_ROOT")
MRNET_ARCHIVE_PATH = os.environ.get("THOR_MRNET_ARCHIVE_PATH")
# MRNet generally requires accepting Stanford AIMI/Redivis terms before download.
# Leave the default empty and let callers pass a signed URL if they have one.
MRNET_DATASET_URL = os.environ.get("THOR_MRNET_DATASET_URL", "")
MRNET_PLANE = os.environ.get("THOR_MRNET_PLANE", "axial")
MRNET_DEPTH = int(os.environ.get("THOR_MRNET_DEPTH", "32"))
MRNET_IMAGE_SIZE = int(os.environ.get("THOR_MRNET_IMAGE_SIZE", "160"))
MRNET_BATCH_SIZE = int(os.environ.get("THOR_MRNET_BATCH_SIZE", "2"))
MRNET_MAX_TRAIN_EXAMPLES = int(os.environ.get("THOR_MRNET_MAX_TRAIN_EXAMPLES", "24"))
MRNET_MAX_VALIDATE_EXAMPLES = int(os.environ.get("THOR_MRNET_MAX_VALIDATE_EXAMPLES", "8"))
MRNET_ARRAY_VERSION = 1

MSD_BRAIN_CACHE_DIR = Path(os.environ.get("THOR_MSD_BRAIN_CACHE_DIR", "/tmp/thor_msd_braintumour_3d_training"))
MSD_BRAIN_DATASET_URL = os.environ.get(
    "THOR_MSD_BRAIN_DATASET_URL",
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
)
MSD_BRAIN_PATCH_SIZE = int(os.environ.get("THOR_MSD_BRAIN_PATCH_SIZE", "64"))
MSD_BRAIN_BATCH_SIZE = int(os.environ.get("THOR_MSD_BRAIN_BATCH_SIZE", "1"))
MSD_BRAIN_MAX_TRAIN_EXAMPLES = int(os.environ.get("THOR_MSD_BRAIN_MAX_TRAIN_EXAMPLES", "12"))
MSD_BRAIN_MAX_VALIDATE_EXAMPLES = int(os.environ.get("THOR_MSD_BRAIN_MAX_VALIDATE_EXAMPLES", "4"))
MSD_BRAIN_ARRAY_VERSION = 1

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.mri_3d_integration,
    pytest.mark.skipif(
        not RUN_MRI_3D_INTEGRATION,
        reason=integration_skip_reason(
            "THOR_RUN_TRAINING_MRI_3D_INTEGRATION",
            description="heavyweight static-volume MRI 3D training tests",
        ),
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+)"
)
_MRNET_LABEL_TASKS = ("abnormal", "acl", "meniscus")
_MSD_NIFTI_DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


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
            capture_file = tempfile.NamedTemporaryFile(prefix=f"thor_mri_3d_fit_fd{fd}_", suffix=".log", delete=False)
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
            }
        )
    return stats


def _fit_and_capture_stats(trainer, *, epochs: int):
    with _NativeOutputTee() as tee:
        trainer.fit(epochs=epochs)
    captured_text = tee.text()
    stats = _captured_trainer_stats(captured_text)

    assert stats, "trainer emitted no parseable stats; set THOR_MRI_3D_STATS_INTERVAL_S=0.0 if this happens"
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
        assert loss >= 0.0, f"{model_name}: negative loss reported: {loss}; phase_counts={phase_counts}"
    assert any(entry["phase"] == "train" for entry in stats), f"{model_name}: no train stats reported; phase_counts={phase_counts}"
    assert any(entry["phase"] == "validate" for entry in stats), f"{model_name}: no validate stats reported; phase_counts={phase_counts}"


def _safe_extract_tar(archive: tarfile.TarFile, dest_dir: Path):
    dest_resolved = dest_dir.resolve()
    for member in archive.getmembers():
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise RuntimeError(f"refusing to extract path outside cache directory: {member.name}")
    archive.extractall(dest_resolved)


def _safe_extract_zip(archive_path: Path, dest_dir: Path):
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (dest_resolved / member.filename).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise RuntimeError(f"refusing to extract path outside cache directory: {member.filename}")
        archive.extractall(dest_resolved)


def _download_file(url: str, archive_path: Path):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(archive_path)


def _ensure_mrnet_root() -> Path:
    if MRNET_ROOT:
        root = Path(MRNET_ROOT).expanduser().resolve()
        if not root.exists():
            pytest.skip(f"THOR_MRNET_ROOT does not exist: {root}")
        if root.name != "MRNet-v1.0" and (root / "MRNet-v1.0").exists():
            root = root / "MRNet-v1.0"
        return root

    MRNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root = MRNET_CACHE_DIR / "MRNet-v1.0"
    if root.exists() and not MRI_3D_REBUILD:
        return root

    if MRI_3D_REBUILD and root.exists():
        shutil.rmtree(root)

    archive_path = Path(MRNET_ARCHIVE_PATH).expanduser().resolve() if MRNET_ARCHIVE_PATH else MRNET_CACHE_DIR / "MRNet-v1.0.zip"
    if not archive_path.exists():
        if not MRNET_DATASET_URL:
            pytest.skip(
                "MRNet is public research data but requires access/terms acceptance before download. "
                "Set THOR_MRNET_ROOT to an extracted MRNet-v1.0 directory, THOR_MRNET_ARCHIVE_PATH to MRNet-v1.0.zip, "
                "or THOR_MRNET_DATASET_URL to a signed download URL."
            )
        _download_file(MRNET_DATASET_URL, archive_path)

    _safe_extract_zip(archive_path, MRNET_CACHE_DIR)
    if not root.exists():
        candidates = [path for path in MRNET_CACHE_DIR.iterdir() if path.is_dir() and (path / "train").exists() and (path / "valid").exists()]
        if len(candidates) == 1:
            root = candidates[0]
        else:
            raise RuntimeError(f"MRNet extraction did not create expected directory {root}")
    return root


def _resize_slice(slice_2d: np.ndarray, *, image_size: int) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        pytest.skip("MRNet preprocessing requires Pillow in the test venv")
        raise AssertionError("unreachable") from exc

    image = np.asarray(slice_2d, dtype=np.float32)
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    uint8_image = np.asarray(np.rint(image * 255.0), dtype=np.uint8)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = Image.fromarray(uint8_image, mode="L").resize((image_size, image_size), resampling)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _uniform_indices(num_items: int, num_samples: int) -> np.ndarray:
    if num_items <= 0:
        raise ValueError("cannot sample an empty volume")
    if num_items == 1:
        return np.zeros((num_samples,), dtype=np.int64)
    return np.rint(np.linspace(0, num_items - 1, num_samples)).astype(np.int64)


def _load_mrnet_volume_cthw(path: Path, *, depth: int, image_size: int) -> np.ndarray:
    volume = np.load(path)
    if volume.ndim != 3:
        raise RuntimeError(f"expected MRNet .npy volume with shape [slices, H, W], got {volume.shape} from {path}")
    selected = _uniform_indices(volume.shape[0], depth)
    slices = [_resize_slice(volume[int(index)], image_size=image_size) for index in selected]
    d_h_w = np.stack(slices, axis=0).astype(np.float32)
    mean = float(d_h_w.mean())
    std = float(d_h_w.std())
    if std < 1.0e-6:
        std = 1.0
    d_h_w = np.clip((d_h_w - mean) / std, -6.0, 6.0)
    return np.ascontiguousarray(d_h_w[None, :, :, :], dtype=np.float16)


def _read_mrnet_split_labels(root: Path, split_name: str) -> dict[int, np.ndarray]:
    labels_by_case: dict[int, list[float]] = {}
    for task_index, task_name in enumerate(_MRNET_LABEL_TASKS):
        csv_path = root / f"{split_name}-{task_name}.csv"
        if not csv_path.exists():
            raise RuntimeError(f"MRNet label file missing: {csv_path}")
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    case_id = int(row[0])
                    value = float(row[1])
                except ValueError:
                    continue
                if case_id not in labels_by_case:
                    labels_by_case[case_id] = [0.0, 0.0, 0.0]
                labels_by_case[case_id][task_index] = value
    return {case_id: np.asarray(values, dtype=np.float16) for case_id, values in labels_by_case.items()}


def _mrnet_array_cache_path() -> Path:
    return MRNET_CACHE_DIR / (
        f"mrnet_{MRNET_PLANE}_{MRNET_DEPTH}x{MRNET_IMAGE_SIZE}_"
        f"train{MRNET_MAX_TRAIN_EXAMPLES}_val{MRNET_MAX_VALIDATE_EXAMPLES}_fp16_v{MRNET_ARRAY_VERSION}.npz"
    )


def _read_mrnet_cached_arrays():
    cache_path = _mrnet_array_cache_path()
    if MRI_3D_REBUILD or not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"].item()))
        expected = {
            "version": MRNET_ARRAY_VERSION,
            "plane": MRNET_PLANE,
            "depth": MRNET_DEPTH,
            "image_size": MRNET_IMAGE_SIZE,
            "max_train_examples": MRNET_MAX_TRAIN_EXAMPLES,
            "max_validate_examples": MRNET_MAX_VALIDATE_EXAMPLES,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        return (
            np.ascontiguousarray(data["train_examples"], dtype=np.float16),
            np.ascontiguousarray(data["train_labels"], dtype=np.float16),
            np.ascontiguousarray(data["validate_examples"], dtype=np.float16),
            np.ascontiguousarray(data["validate_labels"], dtype=np.float16),
            manifest,
        )


def _collect_mrnet_split(root: Path, split_name: str, *, max_examples: int):
    if MRNET_PLANE not in ("axial", "coronal", "sagittal"):
        raise RuntimeError("THOR_MRNET_PLANE must be one of axial, coronal, sagittal")
    labels = _read_mrnet_split_labels(root, split_name)
    plane_dir = root / split_name / MRNET_PLANE
    paths = sorted(path for path in plane_dir.glob("*.npy") if path.is_file())
    if max_examples > 0:
        paths = paths[:max_examples]
    examples = []
    label_rows = []
    for path in paths:
        try:
            case_id = int(path.stem)
        except ValueError:
            continue
        if case_id not in labels:
            continue
        examples.append(_load_mrnet_volume_cthw(path, depth=MRNET_DEPTH, image_size=MRNET_IMAGE_SIZE))
        label_rows.append(labels[case_id])
    if not examples:
        raise RuntimeError(f"MRNet split {split_name} produced no usable {MRNET_PLANE} volumes")
    return np.stack(examples, axis=0), np.stack(label_rows, axis=0)


def _ensure_mrnet_arrays():
    MRNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _read_mrnet_cached_arrays()
    if cached is not None:
        return cached

    root = _ensure_mrnet_root()
    train_examples, train_labels = _collect_mrnet_split(root, "train", max_examples=MRNET_MAX_TRAIN_EXAMPLES)
    validate_examples, validate_labels = _collect_mrnet_split(root, "valid", max_examples=MRNET_MAX_VALIDATE_EXAMPLES)
    manifest = {
        "version": MRNET_ARRAY_VERSION,
        "dataset_root": str(root),
        "plane": MRNET_PLANE,
        "depth": MRNET_DEPTH,
        "image_size": MRNET_IMAGE_SIZE,
        "max_train_examples": MRNET_MAX_TRAIN_EXAMPLES,
        "max_validate_examples": MRNET_MAX_VALIDATE_EXAMPLES,
        "train_examples": int(train_examples.shape[0]),
        "validate_examples": int(validate_examples.shape[0]),
        "example_shape": [1, MRNET_DEPTH, MRNET_IMAGE_SIZE, MRNET_IMAGE_SIZE],
        "label_shape": [len(_MRNET_LABEL_TASKS)],
        "label_names": list(_MRNET_LABEL_TASKS),
    }
    np.savez(
        _mrnet_array_cache_path(),
        train_examples=np.ascontiguousarray(train_examples, dtype=np.float16),
        train_labels=np.ascontiguousarray(train_labels, dtype=np.float16),
        validate_examples=np.ascontiguousarray(validate_examples, dtype=np.float16),
        validate_labels=np.ascontiguousarray(validate_labels, dtype=np.float16),
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return train_examples, train_labels, validate_examples, validate_labels, manifest


def _mrnet_loader(*, batch_size: int):
    train_examples, train_labels, validate_examples, validate_labels, manifest = _ensure_mrnet_arrays()
    loader = thor.training.NumpyFloat16BatchLoader(
        train_examples,
        train_labels,
        validate_examples,
        validate_labels,
        batch_size=batch_size,
        dataset_name="mrnet_axial_volume_fp16",
    )
    return loader, manifest


def _ensure_msd_brain_root() -> Path:
    MSD_BRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root = MSD_BRAIN_CACHE_DIR / "Task01_BrainTumour"
    if root.exists() and not MRI_3D_REBUILD:
        return root
    if MRI_3D_REBUILD and root.exists():
        shutil.rmtree(root)

    archive_path = MSD_BRAIN_CACHE_DIR / "Task01_BrainTumour.tar"
    if not archive_path.exists() or MRI_3D_REBUILD:
        _download_file(MSD_BRAIN_DATASET_URL, archive_path)

    with tarfile.open(archive_path, "r") as archive:
        _safe_extract_tar(archive, MSD_BRAIN_CACHE_DIR)
    if not root.exists():
        raise RuntimeError(f"MSD BrainTumour extraction did not create expected directory {root}")
    return root


def _read_nifti(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    if len(payload) < 352:
        raise RuntimeError(f"NIfTI file too small: {path}")

    sizeof_hdr_le = int(np.frombuffer(payload, dtype="<i4", count=1, offset=0)[0])
    sizeof_hdr_be = int(np.frombuffer(payload, dtype=">i4", count=1, offset=0)[0])
    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        raise RuntimeError(f"Unsupported NIfTI header size for {path}: le={sizeof_hdr_le} be={sizeof_hdr_be}")

    dim = np.frombuffer(payload, dtype=f"{endian}i2", count=8, offset=40).astype(np.int64)
    ndim = int(dim[0])
    if ndim <= 0 or ndim > 7:
        raise RuntimeError(f"Unsupported NIfTI rank {ndim} for {path}")
    shape = tuple(int(value) for value in dim[1 : ndim + 1])
    datatype = int(np.frombuffer(payload, dtype=f"{endian}i2", count=1, offset=70)[0])
    dtype = _MSD_NIFTI_DTYPE_MAP.get(datatype)
    if dtype is None:
        raise RuntimeError(f"Unsupported NIfTI datatype code {datatype} for {path}")
    dtype = np.dtype(dtype).newbyteorder(endian)
    vox_offset = int(float(np.frombuffer(payload, dtype=f"{endian}f4", count=1, offset=108)[0]))
    scl_slope = float(np.frombuffer(payload, dtype=f"{endian}f4", count=1, offset=112)[0])
    scl_inter = float(np.frombuffer(payload, dtype=f"{endian}f4", count=1, offset=116)[0])
    if not math.isfinite(scl_slope) or scl_slope == 0.0:
        scl_slope = 1.0
    if not math.isfinite(scl_inter):
        scl_inter = 0.0
    count = int(np.prod(shape))
    array = np.frombuffer(payload, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")
    array = np.asarray(array)
    if scl_slope != 1.0 or scl_inter != 0.0:
        array = array.astype(np.float32) * scl_slope + scl_inter
    return array


def _crop_start(center: int, patch_size: int, extent: int) -> int:
    if patch_size > extent:
        raise RuntimeError(f"patch size {patch_size} is larger than volume extent {extent}")
    return max(0, min(int(center) - patch_size // 2, extent - patch_size))


def _normalize_msd_channel(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    if not np.isfinite(channel).all():
        channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
    foreground = channel[np.abs(channel) > 1.0e-6]
    values = foreground if foreground.size > 32 else channel.reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1.0e-6:
        std = 1.0
    return np.clip((channel - mean) / std, -6.0, 6.0)


def _load_msd_brain_patch(image_path: Path, label_path: Path, *, patch_size: int):
    image = _read_nifti(image_path)
    label = _read_nifti(label_path)
    if image.ndim != 4 or image.shape[-1] != 4:
        raise RuntimeError(f"expected MSD BrainTumour image shape [X, Y, Z, 4], got {image.shape} from {image_path}")
    if label.ndim != 3:
        raise RuntimeError(f"expected MSD BrainTumour label shape [X, Y, Z], got {label.shape} from {label_path}")
    if image.shape[:3] != label.shape:
        raise RuntimeError(f"image/label shape mismatch for {image_path}: {image.shape} vs {label.shape}")

    tumor_voxels = np.argwhere(label > 0)
    if tumor_voxels.size:
        center = np.rint(tumor_voxels.mean(axis=0)).astype(np.int64)
    else:
        center = np.asarray([extent // 2 for extent in label.shape], dtype=np.int64)
    starts = [_crop_start(int(center[axis]), patch_size, int(label.shape[axis])) for axis in range(3)]
    xs, ys, zs = starts
    xe, ye, ze = xs + patch_size, ys + patch_size, zs + patch_size

    image_patch = image[xs:xe, ys:ye, zs:ze, :]
    label_patch = (label[xs:xe, ys:ye, zs:ze] > 0).astype(np.float32)
    if image_patch.shape != (patch_size, patch_size, patch_size, 4):
        raise RuntimeError(f"unexpected MSD patch shape {image_patch.shape} from {image_path}")

    channels = []
    for channel_index in range(4):
        # Convert [X, Y, Z] to Thor CDHW with D=Z, H=Y, W=X.
        channel = _normalize_msd_channel(image_patch[:, :, :, channel_index]).transpose(2, 1, 0)
        channels.append(channel)
    image_cthw = np.stack(channels, axis=0)
    mask_cthw = label_patch.transpose(2, 1, 0)[None, :, :, :]
    return np.ascontiguousarray(image_cthw, dtype=np.float16), np.ascontiguousarray(mask_cthw, dtype=np.float16)


def _msd_brain_array_cache_path() -> Path:
    return MSD_BRAIN_CACHE_DIR / (
        f"msd_braintumour_patch{MSD_BRAIN_PATCH_SIZE}_train{MSD_BRAIN_MAX_TRAIN_EXAMPLES}_"
        f"val{MSD_BRAIN_MAX_VALIDATE_EXAMPLES}_fp16_v{MSD_BRAIN_ARRAY_VERSION}.npz"
    )


def _read_msd_brain_cached_arrays():
    cache_path = _msd_brain_array_cache_path()
    if MRI_3D_REBUILD or not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"].item()))
        expected = {
            "version": MSD_BRAIN_ARRAY_VERSION,
            "dataset_url": MSD_BRAIN_DATASET_URL,
            "patch_size": MSD_BRAIN_PATCH_SIZE,
            "max_train_examples": MSD_BRAIN_MAX_TRAIN_EXAMPLES,
            "max_validate_examples": MSD_BRAIN_MAX_VALIDATE_EXAMPLES,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        return (
            np.ascontiguousarray(data["train_examples"], dtype=np.float16),
            np.ascontiguousarray(data["train_labels"], dtype=np.float16),
            np.ascontiguousarray(data["validate_examples"], dtype=np.float16),
            np.ascontiguousarray(data["validate_labels"], dtype=np.float16),
            manifest,
        )


def _ensure_msd_brain_arrays():
    MSD_BRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _read_msd_brain_cached_arrays()
    if cached is not None:
        return cached

    root = _ensure_msd_brain_root()
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"
    image_paths = sorted(path for path in images_dir.glob("*.nii.gz") if path.is_file())
    if len(image_paths) < 2:
        raise RuntimeError(f"MSD BrainTumour imagesTr has too few images under {images_dir}")

    train_limit = MSD_BRAIN_MAX_TRAIN_EXAMPLES if MSD_BRAIN_MAX_TRAIN_EXAMPLES > 0 else max(1, len(image_paths) - MSD_BRAIN_MAX_VALIDATE_EXAMPLES)
    validate_limit = MSD_BRAIN_MAX_VALIDATE_EXAMPLES if MSD_BRAIN_MAX_VALIDATE_EXAMPLES > 0 else max(1, len(image_paths) // 10)
    selected_train = image_paths[:train_limit]
    selected_validate = image_paths[train_limit : train_limit + validate_limit]
    if not selected_validate:
        selected_validate = image_paths[-validate_limit:]

    def build_split(paths):
        examples = []
        labels = []
        for image_path in paths:
            label_path = labels_dir / image_path.name
            if not label_path.exists():
                raise RuntimeError(f"MSD BrainTumour label missing for {image_path}: {label_path}")
            example, label = _load_msd_brain_patch(image_path, label_path, patch_size=MSD_BRAIN_PATCH_SIZE)
            examples.append(example)
            labels.append(label)
        if not examples:
            raise RuntimeError("MSD BrainTumour split produced no usable patches")
        return np.stack(examples, axis=0), np.stack(labels, axis=0)

    train_examples, train_labels = build_split(selected_train)
    validate_examples, validate_labels = build_split(selected_validate)
    manifest = {
        "version": MSD_BRAIN_ARRAY_VERSION,
        "dataset_url": MSD_BRAIN_DATASET_URL,
        "dataset_root": str(root),
        "patch_size": MSD_BRAIN_PATCH_SIZE,
        "max_train_examples": MSD_BRAIN_MAX_TRAIN_EXAMPLES,
        "max_validate_examples": MSD_BRAIN_MAX_VALIDATE_EXAMPLES,
        "train_examples": int(train_examples.shape[0]),
        "validate_examples": int(validate_examples.shape[0]),
        "example_shape": [4, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE],
        "label_shape": [1, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE],
        "modalities": ["FLAIR", "T1w", "T1gd", "T2w"],
        "target": "binary_tumor_mask_from_msd_labels_gt_0",
    }
    np.savez(
        _msd_brain_array_cache_path(),
        train_examples=np.ascontiguousarray(train_examples, dtype=np.float16),
        train_labels=np.ascontiguousarray(train_labels, dtype=np.float16),
        validate_examples=np.ascontiguousarray(validate_examples, dtype=np.float16),
        validate_labels=np.ascontiguousarray(validate_labels, dtype=np.float16),
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return train_examples, train_labels, validate_examples, validate_labels, manifest


def _msd_brain_loader(*, batch_size: int):
    train_examples, train_labels, validate_examples, validate_labels, manifest = _ensure_msd_brain_arrays()
    loader = thor.training.NumpyFloat16BatchLoader(
        train_examples,
        train_labels,
        validate_examples,
        validate_labels,
        batch_size=batch_size,
        dataset_name="msd_braintumour_patch_segmentation_fp16",
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


class _VectorElement(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, vector: thor.Tensor, *, index: int, output_dtype=thor.DataType.fp16):
        dims = vector.get_dimensions()
        assert len(dims) == 1
        assert 0 <= index < dims[0]
        self.index = index
        self.vector_dims = dims[0]
        self.output_dtype = output_dtype
        super().__init__(network=network, inputs=vector, output_names=["value"])

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        vector_tensor = context.input_tensor("feature_input")
        dims = vector_tensor.get_dimensions()
        assert len(dims) == 2
        batch_size, vector_dims = dims
        assert vector_dims == self.vector_dims
        vector = context.input("feature_input")
        value = vector.strided_view([batch_size, 1], [vector_dims, 1], self.index)
        return {"value": value.with_dtypes(output_dtype=self.output_dtype, compute_dtype=thor.DataType.fp32)}


def _vector_element(network: thor.Network, vector: thor.Tensor, index: int, *, output_dtype=thor.DataType.fp16) -> thor.Tensor:
    return _VectorElement(network, vector, index=index, output_dtype=output_dtype)["value"]


def _build_mrnet_axial_conv3d_classifier(
    name: str,
    *,
    depth: int,
    image_size: int,
    dtype=thor.DataType.fp16,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [1, depth, image_size, image_size], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [len(_MRNET_LABEL_TASKS)], dtype)

    x = examples.get_feature_output()
    x = _conv3d_bn_relu(
        network,
        x,
        16,
        3,
        7,
        7,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=3,
        horizontal_padding=3,
        dtype=dtype,
    )
    x = _conv3d_bn_relu(
        network,
        x,
        32,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype,
    )
    x = _conv3d_bn_relu(
        network,
        x,
        64,
        3,
        3,
        3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        dtype=dtype,
    )
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
        dtype=dtype,
    )
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
        dtype=dtype,
    )
    # Keep the explicit Flatten here as a real 3D integration regression for
    # Flatten's metadata-only forward alias and shape-restoring backward alias.
    flat = thor.layers.Flatten(network, x, 1)
    hidden = thor.layers.FullyConnected(network, flat.get_feature_output(), 1024, True, activation=thor.activations.Relu())
    x = hidden.get_feature_output()
    drop = thor.layers.DropOut(network, x, 0.25)
    predictions = thor.layers.FullyConnected(network, drop.get_feature_output(), len(_MRNET_LABEL_TASKS), True, activation=thor.activations.Sigmoid())

    abnormal_prediction = _vector_element(network, predictions.get_feature_output(), 0, output_dtype=dtype)
    abnormal_label = _vector_element(network, labels.get_feature_output(), 0, output_dtype=dtype)
    loss = thor.losses.BinaryCrossEntropy(network, abnormal_prediction, abnormal_label, thor.DataType.fp32)

    # Keep the ACL/meniscus outputs visible so the test exercises a real multi-label
    # head shape even though the scalar training loss is the abnormality BCE.
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "predictions", predictions.get_feature_output(), dtype)
    return network


def _build_msd_brain_3d_segmentation_network(
    name: str,
    *,
    patch_size: int,
    dtype=thor.DataType.fp16,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [4, patch_size, patch_size, patch_size], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [1, patch_size, patch_size, patch_size], dtype)

    x = examples.get_feature_output()
    x = _conv3d_bn_relu(network, x, 16, 3, 3, 3, depth_padding=1, vertical_padding=1, horizontal_padding=1, dtype=dtype)
    x = _conv3d_bn_relu(network, x, 32, 3, 3, 3, depth_padding=1, vertical_padding=1, horizontal_padding=1, dtype=dtype)
    x = _conv3d_bn_relu(network, x, 32, 3, 3, 3, depth_padding=1, vertical_padding=1, horizontal_padding=1, dtype=dtype)
    x = _conv3d_bn_relu(network, x, 16, 3, 3, 3, depth_padding=1, vertical_padding=1, horizontal_padding=1, dtype=dtype)
    mask_probs = thor.layers.Convolution3d(
        network,
        x,
        1,
        3,
        3,
        3,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=thor.activations.Sigmoid(),
    )

    loss = thor.losses.segmentation.DiceLoss(
        network,
        mask_probs.get_feature_output(),
        labels.get_feature_output(),
        1.0,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "mask_probs", mask_probs.get_feature_output(), dtype)
    return network


def _run_mrnet_training(capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _mrnet_loader(batch_size=MRNET_BATCH_SIZE)
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "MRNet train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "MRNet validation split unexpectedly has zero batches"
        assert manifest["example_shape"] == [1, MRNET_DEPTH, MRNET_IMAGE_SIZE, MRNET_IMAGE_SIZE]
        assert manifest["label_shape"] == [len(_MRNET_LABEL_TASKS)]

        network = _build_mrnet_axial_conv3d_classifier(
            "python_integration_mrnet_static_3d_knee_classifier",
            depth=manifest["depth"],
            image_size=manifest["image_size"],
        )
        optimizer = thor.optimizers.AdamW(alpha=0.0003, beta1=0.9, beta2=0.999, weight_decay=0.01)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats_interval_s=MRI_3D_STATS_INTERVAL_S,
            max_in_flight_batches=MRI_3D_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=MRI_3D_EPOCHS)
        _assert_finite_positive_losses(stats, model_name="mrnet_static_3d_knee_classifier")


def _run_msd_brain_training(capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _msd_brain_loader(batch_size=MSD_BRAIN_BATCH_SIZE)
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "MSD BrainTumour train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "MSD BrainTumour validation split unexpectedly has zero batches"
        assert manifest["example_shape"] == [4, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE]
        assert manifest["label_shape"] == [1, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE, MSD_BRAIN_PATCH_SIZE]

        network = _build_msd_brain_3d_segmentation_network(
            "python_integration_msd_braintumour_3d_patch_segmenter",
            patch_size=manifest["patch_size"],
        )
        optimizer = thor.optimizers.AdamW(alpha=0.0005, beta1=0.9, beta2=0.999, weight_decay=0.01)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats_interval_s=MRI_3D_STATS_INTERVAL_S,
            max_in_flight_batches=MRI_3D_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=MRI_3D_EPOCHS)
        _assert_finite_positive_losses(stats, model_name="msd_braintumour_3d_patch_segmenter")


def test_queued_trainer_trains_mrnet_static_3d_knee_classifier_end_to_end(capfd):
    _run_mrnet_training(capfd)


def test_queued_trainer_trains_msd_braintumour_3d_patch_segmenter_end_to_end(capfd):
    _run_msd_brain_training(capfd)
