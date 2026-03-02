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


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda") and not has_cuda_gpu():
        pytest.skip("CUDA GPU not available")
