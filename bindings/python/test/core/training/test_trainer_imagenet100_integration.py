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

RUN_IMAGENET100_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_IMAGENET100_INTEGRATION") == "1"
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
IMAGENET100_MANIFEST_VERSION = 1
IMAGENET100_NUM_CLASSES = 100
IMAGENET100_TRAIN_EXAMPLES = 126_689
IMAGENET100_VALIDATE_EXAMPLES = 5_000

OBJECT_DETECTION_CACHE_DIR = Path(os.environ.get("THOR_OBJECT_DETECTION_CACHE_DIR", "/tmp/thor_voc_detection_training"))
OBJECT_DETECTION_DATASET_URL = os.environ.get(
    "THOR_OBJECT_DETECTION_DATASET_URL",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
)
OBJECT_DETECTION_IMAGE_SIZE = int(os.environ.get("THOR_OBJECT_DETECTION_IMAGE_SIZE", "224"))
OBJECT_DETECTION_BATCH_SIZE = int(os.environ.get("THOR_OBJECT_DETECTION_BATCH_SIZE", str(IMAGENET100_BATCH_SIZE)))
OBJECT_DETECTION_EPOCHS = int(os.environ.get("THOR_OBJECT_DETECTION_EPOCHS", str(IMAGENET100_EPOCHS)))
OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES = int(
    os.environ.get("THOR_OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES", str(IMAGENET100_MAX_IN_FLIGHT_BATCHES))
)
OBJECT_DETECTION_STATS_INTERVAL_S = float(
    os.environ.get("THOR_OBJECT_DETECTION_STATS_INTERVAL_S", str(IMAGENET100_STATS_INTERVAL_S))
)
OBJECT_DETECTION_REBUILD = os.environ.get("THOR_OBJECT_DETECTION_REBUILD") == "1"
OBJECT_DETECTION_NUM_SHARDS = int(os.environ.get("THOR_OBJECT_DETECTION_NUM_SHARDS", "1"))
OBJECT_DETECTION_MANIFEST_VERSION = 1
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
        not RUN_IMAGENET100_INTEGRATION,
        reason="set THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1 to run heavyweight ImageNet-100 model training tests",
    ),
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAINER_STATS_RE = re.compile(
    r"INFO trainer: phase=(?P<phase>train|validate|test) "
    r"epoch=(?P<epoch>\d+)/(?:\d+) "
    r"step=(?P<step>\d+) "
    r"batch=(?P<batch>\d+)/(?:\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+)")


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
            "before setting THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1.") from exc
    return load_dataset, Image


def _class_dir(label: int) -> str:
    return f"class_{label:03d}"


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


def _prepare_split(ds, *, split_name: str, raw_root: Path, num_classes: int, image_size: int, resize_shorter_side: int):
    for label in range(num_classes):
        (raw_root / split_name / _class_dir(label)).mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(ds):
        label = int(example["label"])
        if label < 0 or label >= num_classes:
            raise RuntimeError(f"{split_name}: label {label} is outside [0, {num_classes})")
        processed = _center_crop_resize_to_chw_fp16(
            example["image"],
            image_size=image_size,
            resize_shorter_side=resize_shorter_side,
        )
        filename = raw_root / split_name / _class_dir(label) / f"{split_name}_{index:08d}.bin"
        filename.write_bytes(processed.tobytes(order="C"))


def _mirror_validate_as_test(raw_root: Path, *, num_classes: int):
    for label in range(num_classes):
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


def _manifest_path(cache_root: Path) -> Path:
    return cache_root / f"imagenet100_{IMAGENET100_IMAGE_SIZE}_fp16_manifest.json"


def _shard_root(cache_root: Path) -> Path:
    return cache_root / f"shards_{IMAGENET100_IMAGE_SIZE}_fp16"


def _base_manifest(*, shard_paths):
    return {
        "version": IMAGENET100_MANIFEST_VERSION,
        "dataset_id": IMAGENET100_DATASET_ID,
        "image_size": IMAGENET100_IMAGE_SIZE,
        "resize_shorter_side": IMAGENET100_RESIZE_SHORTER_SIDE,
        "dtype": "fp16",
        "num_classes": IMAGENET100_NUM_CLASSES,
        "train_examples": IMAGENET100_TRAIN_EXAMPLES,
        "validate_examples": IMAGENET100_VALIDATE_EXAMPLES,
        "test_examples": IMAGENET100_VALIDATE_EXAMPLES,
        "example_shape": [3, IMAGENET100_IMAGE_SIZE, IMAGENET100_IMAGE_SIZE],
        "label_shape": [IMAGENET100_NUM_CLASSES],
        "shard_paths": sorted(str(Path(path)) for path in shard_paths),
        "label_names": [],
    }


def _expected_shard_paths(cache_root: Path):
    shard_root = _shard_root(cache_root)
    base_name = f"imagenet100_{IMAGENET100_IMAGE_SIZE}_fp16_chw"
    return [
        shard_root / f"dest_{shard_index:02d}" / f"{base_name}_{shard_index + 1}_of_{IMAGENET100_NUM_SHARDS}.shard"
        for shard_index in range(IMAGENET100_NUM_SHARDS)
    ]


def _existing_shard_paths(cache_root: Path):
    if IMAGENET100_REBUILD:
        return []

    # Treat the finalized shard filename as the cache contract. Do not scan,
    # validate, load the Hugging Face dataset, inspect raw preprocessing output,
    # or touch the shard directory on the reuse path. If this file is present,
    # assume the cache is good; delete it or set THOR_IMAGENET100_REBUILD=1 to
    # force regeneration.
    expected_paths = _expected_shard_paths(cache_root)
    if all(path.exists() for path in expected_paths):
        return [str(path) for path in expected_paths]
    return []


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
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    if not manifest.get("shard_paths"):
        return None
    # Do not stat/validate the completed shard set here. These heavyweight tests
    # intentionally treat an existing finalized shard cache as authoritative; if
    # it is bad, delete it or set THOR_IMAGENET100_REBUILD=1. The fast path must
    # not touch or rewrite /tmp/thor_imagenet100_training/shards_*/dest_*.
    return manifest


def _ensure_imagenet100_shards():
    IMAGENET100_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing_shards = _existing_shard_paths(IMAGENET100_CACHE_DIR)
    if existing_shards:
        # The reuse path is intentionally just an expected-file existence check.
        # Do not read/validate metadata, scan directories, import datasets/Pillow,
        # or touch the shard tree before starting the trainer.
        return _base_manifest(shard_paths=existing_shards)

    manifest = _read_manifest_if_valid(IMAGENET100_CACHE_DIR)
    if manifest is not None:
        return manifest

    load_dataset, _ = _import_imagenet_dependencies()
    processing_root = IMAGENET100_CACHE_DIR / "processing_tmp"
    raw_root = processing_root / "raw_fp16_chw"
    shard_root = _shard_root(IMAGENET100_CACHE_DIR)
    hf_cache = IMAGENET100_CACHE_DIR / "hf_datasets"
    base_name = f"imagenet100_{IMAGENET100_IMAGE_SIZE}_fp16_chw"

    if processing_root.exists():
        shutil.rmtree(processing_root)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_dest_dirs = []
    for shard_index in range(IMAGENET100_NUM_SHARDS):
        dest = shard_root / f"dest_{shard_index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs.append(dest)

    train = load_dataset(IMAGENET100_DATASET_ID, split="train", cache_dir=str(hf_cache))
    validate = load_dataset(IMAGENET100_DATASET_ID, split="validation", cache_dir=str(hf_cache))
    label_feature = train.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    num_classes = len(label_names) if label_names else int(max(train["label"])) + 1

    _prepare_split(
        train,
        split_name="train",
        raw_root=raw_root,
        num_classes=num_classes,
        image_size=IMAGENET100_IMAGE_SIZE,
        resize_shorter_side=IMAGENET100_RESIZE_SHORTER_SIDE,
    )
    _prepare_split(
        validate,
        split_name="validate",
        raw_root=raw_root,
        num_classes=num_classes,
        image_size=IMAGENET100_IMAGE_SIZE,
        resize_shorter_side=IMAGENET100_RESIZE_SHORTER_SIDE,
    )
    # The Hugging Face dataset exposes train/validation. Thor's shard loader also
    # opens a TEST assembler, so use the fully preprocessed validation payload as
    # the test split too. Training and validation still consume the full official
    # train and validation splits.
    _mirror_validate_as_test(raw_root, num_classes=num_classes)
    example_size_in_bytes = 3 * IMAGENET100_IMAGE_SIZE * IMAGENET100_IMAGE_SIZE * np.dtype(np.float16).itemsize
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

    manifest = _base_manifest(shard_paths=shard_paths)
    manifest.update(
        {
            "num_classes": num_classes,
            "train_examples": len(train),
            "validate_examples": len(validate),
            "test_examples": len(validate),
            "label_shape": [num_classes],
            "label_names": label_names,
        })
    _manifest_path(IMAGENET100_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(processing_root)
    return manifest


def _imagenet100_loader(*, batch_size: int):
    manifest = _ensure_imagenet100_shards()
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.fp16,
        manifest["label_shape"],
        thor.DataType.fp16,
        batch_size=batch_size,
        dataset_name="clane9_imagenet100_preprocessed_fp16_chw",
    )
    return loader, manifest


def _object_detection_image_elems(image_size: int) -> int:
    return 3 * image_size * image_size


def _object_detection_example_elems(image_size: int) -> int:
    return _object_detection_image_elems(image_size) + OBJECT_DETECTION_BOX_DIMS


def _object_detection_manifest_path(cache_root: Path) -> Path:
    return cache_root / f"voc2012_detection_{OBJECT_DETECTION_IMAGE_SIZE}_packed_fp16_manifest.json"


def _object_detection_shard_root(cache_root: Path) -> Path:
    return cache_root / f"voc2012_detection_shards_{OBJECT_DETECTION_IMAGE_SIZE}_packed_fp16"


def _object_detection_base_manifest(*, shard_paths, train_examples: int, validate_examples: int):
    image_size = OBJECT_DETECTION_IMAGE_SIZE
    return {
        "version": OBJECT_DETECTION_MANIFEST_VERSION,
        "dataset_url": OBJECT_DETECTION_DATASET_URL,
        "image_size": image_size,
        "dtype": "fp16",
        "box_format": "xyxy_normalized",
        "num_classes": len(OBJECT_DETECTION_VOC_CLASSES),
        "train_examples": train_examples,
        "validate_examples": validate_examples,
        "test_examples": validate_examples,
        "example_shape": [_object_detection_example_elems(image_size)],
        "image_shape": [3, image_size, image_size],
        "box_shape": [OBJECT_DETECTION_BOX_DIMS],
        "label_shape": [len(OBJECT_DETECTION_VOC_CLASSES)],
        "shard_paths": sorted(str(Path(path)) for path in shard_paths),
        "label_names": list(OBJECT_DETECTION_VOC_CLASSES),
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
        "label_shape": [len(OBJECT_DETECTION_VOC_CLASSES)],
        "example_shape": [_object_detection_example_elems(OBJECT_DETECTION_IMAGE_SIZE)],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    if not manifest.get("shard_paths"):
        return None
    return manifest


def _import_object_detection_dependencies():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The heavyweight object-detection integration tests require the optional "
            "Python package 'Pillow'. Install it in the test venv before setting "
            "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1."
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


def _ensure_voc_detection_shards():
    OBJECT_DETECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_object_detection_manifest_if_valid(OBJECT_DETECTION_CACHE_DIR)
    if manifest is not None:
        return manifest

    voc_root = _ensure_voc2012_trainval_downloaded(OBJECT_DETECTION_CACHE_DIR)
    processing_root = OBJECT_DETECTION_CACHE_DIR / "processing_tmp"
    raw_root = processing_root / "raw_packed_fp16"
    shard_root = _object_detection_shard_root(OBJECT_DETECTION_CACHE_DIR)
    base_name = f"voc2012_detection_{OBJECT_DETECTION_IMAGE_SIZE}_packed_fp16"

    if processing_root.exists():
        shutil.rmtree(processing_root)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_dest_dirs = []
    for shard_index in range(OBJECT_DETECTION_NUM_SHARDS):
        dest = shard_root / f"dest_{shard_index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        shard_dest_dirs.append(dest)

    train_count = _write_voc_detection_split(
        voc_root,
        split_name="train",
        split_ids=_voc_split_ids(voc_root, "train"),
        raw_root=raw_root,
        image_size=OBJECT_DETECTION_IMAGE_SIZE,
    )
    validate_count = _write_voc_detection_split(
        voc_root,
        split_name="validate",
        split_ids=_voc_split_ids(voc_root, "val"),
        raw_root=raw_root,
        image_size=OBJECT_DETECTION_IMAGE_SIZE,
    )
    _mirror_voc_validate_as_test(raw_root)

    example_size_in_bytes = _object_detection_example_elems(OBJECT_DETECTION_IMAGE_SIZE) * np.dtype(np.float16).itemsize
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

    manifest = _object_detection_base_manifest(
        shard_paths=shard_paths,
        train_examples=train_count,
        validate_examples=validate_count,
    )
    _object_detection_manifest_path(OBJECT_DETECTION_CACHE_DIR).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(processing_root)
    return manifest


def _voc_detection_loader(*, batch_size: int):
    manifest = _ensure_voc_detection_shards()
    loader = thor.training.LocalBatchLoader(
        manifest["shard_paths"],
        manifest["example_shape"],
        thor.DataType.fp16,
        manifest["label_shape"],
        thor.DataType.fp16,
        batch_size=batch_size,
        dataset_name="pascal_voc2012_detection_packed_fp16_chw_xyxy",
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


class _PackedVocDetectionExample(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, packed: thor.Tensor, *, image_size: int):
        self.image_size = image_size
        super().__init__(network=network, inputs=packed, output_names=["image", "box"])

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        packed_tensor = context.input_tensor("feature_input")
        dims = packed_tensor.get_dimensions()
        assert len(dims) == 2
        batch_size, packed_elems = dims
        image_elems = _object_detection_image_elems(self.image_size)
        assert packed_elems == image_elems + OBJECT_DETECTION_BOX_DIMS

        packed = context.input("feature_input")
        image = packed.strided_view(
            [batch_size, 3, self.image_size, self.image_size],
            [packed_elems, self.image_size * self.image_size, self.image_size, 1],
            0,
        )
        box = packed.strided_view([batch_size, OBJECT_DETECTION_BOX_DIMS], [packed_elems, 1], image_elems)
        return {
            "image": (image + 0.0).with_dtypes(output_dtype=thor.DataType.fp16, compute_dtype=thor.DataType.fp32),
            "box": (box + 0.0).with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
        }


class _PackedVocDetectionVideoExample(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, packed: thor.Tensor, *, image_size: int):
        self.image_size = image_size
        super().__init__(network=network, inputs=packed, output_names=["volume", "box"])

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        packed_tensor = context.input_tensor("feature_input")
        dims = packed_tensor.get_dimensions()
        assert len(dims) == 2
        batch_size, packed_elems = dims
        image_elems = _object_detection_image_elems(self.image_size)
        assert packed_elems == image_elems + OBJECT_DETECTION_BOX_DIMS

        packed = context.input("feature_input")
        volume = packed.strided_view(
            [batch_size, 3, 1, self.image_size, self.image_size],
            [packed_elems, self.image_size * self.image_size, self.image_size * self.image_size, self.image_size, 1],
            0,
        )
        box = packed.strided_view([batch_size, OBJECT_DETECTION_BOX_DIMS], [packed_elems, 1], image_elems)
        return {
            "volume": (volume + 0.0).with_dtypes(output_dtype=thor.DataType.fp16, compute_dtype=thor.DataType.fp32),
            "box": (box + 0.0).with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
        }


class _ValidBoxFromUnconstrainedVector(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, raw_box: thor.Tensor):
        assert raw_box.get_dimensions() == [OBJECT_DETECTION_BOX_DIMS]
        super().__init__(network=network, inputs=raw_box, output_names=["x1", "y1", "x2", "y2"])

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        raw_tensor = context.input_tensor("feature_input")
        dims = raw_tensor.get_dimensions()
        assert len(dims) == 2
        batch_size, box_dims = dims
        assert box_dims == OBJECT_DETECTION_BOX_DIMS

        raw = context.input("feature_input")

        def coord(index: int) -> thor.physical.Expression:
            return raw.strided_view([batch_size, 1], [OBJECT_DETECTION_BOX_DIMS, 1], index).with_dtypes(
                output_dtype=thor.DataType.fp32,
                compute_dtype=thor.DataType.fp32,
            )

        sx = thor.physical.Expression.sigmoid(coord(0))
        sy = thor.physical.Expression.sigmoid(coord(1))
        sw = thor.physical.Expression.sigmoid(coord(2))
        sh = thor.physical.Expression.sigmoid(coord(3))

        # Keep the raw head unconstrained for the optimizer, but make the tensor
        # consumed by CIoU/GIoU a valid normalized xyxy box.  Width/height are
        # bounded away from zero so the integration test exercises meaningful box
        # gradients instead of degenerate zero-area boxes.
        x1 = sx * 0.65
        y1 = sy * 0.65
        x2 = x1 + 0.10 + sw * 0.25
        y2 = y1 + 0.10 + sh * 0.25
        return {
            "x1": x1.with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
            "y1": y1.with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
            "x2": x2.with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
            "y2": y2.with_dtypes(output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
        }


def _valid_box_from_raw_head(network: thor.Network, raw_box: thor.Tensor) -> thor.Tensor:
    parts = _ValidBoxFromUnconstrainedVector(network, raw_box)
    box = thor.layers.Concatenate(
        network,
        [parts["x1"], parts["y1"], parts["x2"], parts["y2"]],
        0,
    )
    return box.get_feature_output()


def _split_voc_detection_example(network: thor.Network, packed_examples: thor.Tensor, *, image_size: int):
    split = _PackedVocDetectionExample(network, packed_examples, image_size=image_size)
    return split["image"], split["box"]


def _split_voc_detection_video_example(network: thor.Network, packed_examples: thor.Tensor, *, image_size: int):
    split = _PackedVocDetectionVideoExample(network, packed_examples, image_size=image_size)
    return split["volume"], split["box"]


def _global_average_pool_2d(network: thor.Network, x: thor.Tensor) -> thor.Tensor:
    _, height, width = x.get_dimensions()
    pool = thor.layers.Pooling(network, x, thor.layers.Pooling.Type.average, height, width, 1, 1)
    return pool.get_feature_output()


def _build_tiny_voc2012_multitask_detector(
    name: str,
    *,
    num_classes: int,
    image_size: int = OBJECT_DETECTION_IMAGE_SIZE,
    dtype=thor.DataType.fp16,
):
    network = thor.Network(name)
    packed_examples = thor.layers.NetworkInput(
        network,
        "examples",
        [_object_detection_example_elems(image_size)],
        dtype,
    )
    labels = thor.layers.NetworkInput(network, "labels", [num_classes], dtype)
    image, target_box = _split_voc_detection_example(network, packed_examples.get_feature_output(), image_size=image_size)

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

    class_logits = thor.layers.FullyConnected(network, x, num_classes, True, activation=None)
    raw_box = thor.layers.FullyConnected(network, x, OBJECT_DETECTION_BOX_DIMS, True, activation=None)
    pred_box = _valid_box_from_raw_head(network, raw_box.get_feature_output())

    class_loss = thor.losses.CategoricalCrossEntropy(
        network,
        class_logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    box_loss = thor.losses.detection.CIoULoss(
        network,
        pred_box,
        target_box,
        "xyxy",
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", box_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "class_loss", class_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "pred_boxes", pred_box, thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "class_scores", class_logits.get_feature_output(), dtype)
    return network


def _build_conv3d_voc2012_box_detector(
    name: str,
    *,
    num_classes: int,
    image_size: int = OBJECT_DETECTION_IMAGE_SIZE,
    dtype=thor.DataType.fp16,
):
    del num_classes
    network = thor.Network(name)
    packed_examples = thor.layers.NetworkInput(
        network,
        "examples",
        [_object_detection_example_elems(image_size)],
        dtype,
    )
    volume, target_box = _split_voc_detection_video_example(network, packed_examples.get_feature_output(), image_size=image_size)

    conv1 = thor.layers.Convolution3d(
        network,
        volume,
        16,
        1,
        7,
        7,
        depth_stride=1,
        vertical_stride=4,
        horizontal_stride=4,
        depth_padding=0,
        vertical_padding=3,
        horizontal_padding=3,
        has_bias=True,
        activation=thor.activations.Relu(),
    )
    x = conv1.get_feature_output()
    conv2 = thor.layers.Convolution3d(
        network,
        x,
        32,
        1,
        3,
        3,
        depth_stride=1,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=0,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=thor.activations.Relu(),
    )
    x = conv2.get_feature_output()
    conv3 = thor.layers.Convolution3d(
        network,
        x,
        64,
        1,
        3,
        3,
        depth_stride=1,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=0,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=thor.activations.Relu(),
    )
    x = conv3.get_feature_output()
    flat = thor.layers.Flatten(network, x, 1)
    x = flat.get_feature_output()
    hidden = thor.layers.FullyConnected(network, x, 128, True, activation=thor.activations.Relu())
    raw_box = thor.layers.FullyConnected(network, hidden.get_feature_output(), OBJECT_DETECTION_BOX_DIMS, True, activation=None)
    pred_box = _valid_box_from_raw_head(network, raw_box.get_feature_output())

    loss = thor.losses.detection.GIoULoss(
        network,
        pred_box,
        target_box,
        "xyxy",
        1.0e-5,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "pred_boxes", pred_box, thor.DataType.fp32)
    return network


def _run_full_imagenet100_model_training(model_builder, *, model_name: str, capfd):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        loader, manifest = _imagenet100_loader(batch_size=IMAGENET100_BATCH_SIZE)
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "ImageNet-100 train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "ImageNet-100 validation split unexpectedly has zero batches"
        assert manifest["num_classes"] == manifest["label_shape"][0]

        network = model_builder(f"python_integration_{model_name}_imagenet100_full", num_classes=manifest["num_classes"])
        optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.9)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats=True,
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
        assert manifest["train_examples"] == loader.get_num_train_examples()
        assert manifest["validate_examples"] == loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 0, "VOC2012 detection train split unexpectedly has zero batches"
        assert loader.get_num_validate_batches() > 0, "VOC2012 detection validation split unexpectedly has zero batches"
        assert manifest["num_classes"] == manifest["label_shape"][0]
        assert manifest["box_shape"] == [OBJECT_DETECTION_BOX_DIMS]

        network = model_builder(
            f"python_integration_{model_name}_voc2012_detection_full",
            num_classes=manifest["num_classes"],
            image_size=manifest["image_size"],
        )
        optimizer = thor.optimizers.AdamW(initial_learning_rate=0.001, weight_decay=0.01)
        trainer = thor.training.Trainer(
            network,
            loader,
            optimizer=optimizer,
            debug_synchronous=False,
            stats=True,
            stats_interval_s=OBJECT_DETECTION_STATS_INTERVAL_S,
            max_in_flight_batches=OBJECT_DETECTION_MAX_IN_FLIGHT_BATCHES,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
        )
        stats = _fit_and_capture_stats(trainer, epochs=OBJECT_DETECTION_EPOCHS)
        _assert_finite_positive_losses(stats, model_name=model_name)


@pytest.mark.parametrize(
    ("model_name", "model_builder"),
    [
        ("tiny_multitask_detector", _build_tiny_voc2012_multitask_detector),
        ("conv3d_box_detector", _build_conv3d_voc2012_box_detector),
    ],
)
def test_queued_trainer_trains_voc2012_object_detection_networks_end_to_end(model_name, model_builder, capfd):
    _run_full_voc2012_detection_model_training(model_builder, model_name=model_name, capfd=capfd)


def test_queued_trainer_trains_full_alexnet_on_full_clane9_imagenet100(capfd):
    _run_full_imagenet100_model_training(_build_alexnet_imagenet100, model_name="alexnet", capfd=capfd)


def test_queued_trainer_trains_full_resnet18_on_full_clane9_imagenet100(capfd):
    _run_full_imagenet100_model_training(_build_resnet18_imagenet100, model_name="resnet18", capfd=capfd)
