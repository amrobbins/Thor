import shutil
import subprocess
import pytest
from functools import lru_cache


@lru_cache(maxsize=1)
def has_cuda_gpu() -> bool:
    """CUDA GPU presence check"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        return r.returncode == 0 and "GPU " in (r.stdout or "")
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA-capable GPU")
    config.addinivalue_line("markers", "performance: performance / throughput microbenchmarks")
    config.addinivalue_line("markers", "training_integration: opt-in end-to-end model training tests")
    config.addinivalue_line("markers", "imagenet100_integration: heavyweight ImageNet-100 training integration tests")
    config.addinivalue_line("markers", "ucf101_3d_integration: heavyweight training integration")
    config.addinivalue_line("markers", "digits_dense_integration: heavyweight training integration")
    config.addinivalue_line("markers", "mri_3d_integration: heavyweight training integration")
    config.addinivalue_line("markers", "byte_lm_integration: heavyweight training integration")
    config.addinivalue_line("markers", "digits_dense_cv5_integration: heavyweight training integration")
    config.addinivalue_line("markers", "imagenet100_cv5_integration: heavyweight training integration")


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda") and not has_cuda_gpu():
        pytest.skip("CUDA GPU not available")


def make_numpy_training_data(
    tensors,
    train_indices,
    validate_indices,
    batch_size,
    *,
    test_indices=None,
    randomize_train=False,
    random_seed=None,
    dataset_name="numpy",
    device_storage="off",
):
    """Build immutable NumPy TrainingData from one canonical tensor table."""
    import numpy as np
    import thor

    canonical = {}
    expected_examples = None
    for name, values in tensors.items():
        array = np.asarray(values)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if expected_examples is None:
            expected_examples = int(array.shape[0])
        elif int(array.shape[0]) != expected_examples:
            raise ValueError("all NumPy training tensors must share one leading dimension")
        canonical[name] = array

    dataset = thor.data.NumpyDataset(canonical)
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=train_indices,
        validate_indices=validate_indices,
        test_indices=test_indices,
    )
    policy = thor.data.BatchPolicy(
        batch_size=batch_size,
        randomize_train=randomize_train,
        random_seed=random_seed,
    )
    return thor.data.TrainingData(
        dataset=dataset,
        splits=manifest,
        batching=policy,
        dataset_name=dataset_name,
        device_storage=device_storage,
    )


def make_numpy_training_data_from_splits(
    train,
    validate,
    *,
    batch_size,
    test=None,
    randomize_train=False,
    random_seed=None,
    dataset_name="numpy",
    device_storage="off",
):
    """Combine named split dictionaries into one immutable canonical dataset."""
    import numpy as np

    if set(train) != set(validate):
        raise ValueError("train and validate tensor names must match")
    if test is not None and set(train) != set(test):
        raise ValueError("train and test tensor names must match")

    def split_size(values, split_name):
        sizes = {int(np.asarray(array).shape[0]) for array in values.values()}
        if len(sizes) != 1:
            raise ValueError(f"{split_name} tensors must share one leading dimension")
        return sizes.pop()

    train_size = split_size(train, "train")
    validate_size = split_size(validate, "validate")
    test_size = split_size(test, "test") if test is not None else 0
    tensors = {
        name: np.ascontiguousarray(
            np.concatenate(
                [np.asarray(train[name]), np.asarray(validate[name])]
                + ([np.asarray(test[name])] if test is not None else []),
                axis=0,
            )
        )
        for name in train
    }
    train_indices = np.arange(train_size, dtype=np.int64)
    validate_indices = np.arange(train_size, train_size + validate_size, dtype=np.int64)
    test_indices = (
        np.arange(train_size + validate_size, train_size + validate_size + test_size, dtype=np.int64)
        if test is not None
        else None
    )
    return make_numpy_training_data(
        tensors,
        train_indices,
        validate_indices,
        batch_size,
        test_indices=test_indices,
        randomize_train=randomize_train,
        random_seed=random_seed,
        dataset_name=dataset_name,
        device_storage=device_storage,
    )


def make_numpy_pair_training_data(
    train_examples,
    train_labels,
    validate_examples,
    validate_labels,
    *,
    batch_size,
    example_input_name="examples",
    label_input_name="labels",
    dataset_name="numpy",
    device_storage="off",
):
    return make_numpy_training_data_from_splits(
        {
            example_input_name: train_examples,
            label_input_name: train_labels,
        },
        {
            example_input_name: validate_examples,
            label_input_name: validate_labels,
        },
        batch_size=batch_size,
        randomize_train=False,
        dataset_name=dataset_name,
        device_storage=device_storage,
    )
