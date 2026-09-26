import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_NOFILE import policy is Linux-specific")
def test_import_raises_open_file_soft_limit_to_hard_limit():
    script = r'''
import resource

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if hard == resource.RLIM_INFINITY:
    lowered_soft = 1024
else:
    if hard <= 1:
        raise SystemExit("SKIP_NO_RAISE_RANGE")
    lowered_soft = min(1024, hard - 1)

resource.setrlimit(resource.RLIMIT_NOFILE, (lowered_soft, hard))
before = resource.getrlimit(resource.RLIMIT_NOFILE)
assert before[0] == lowered_soft, before

import thor

after = resource.getrlimit(resource.RLIMIT_NOFILE)
assert after[0] == after[1], (before, after)
assert after[1] == hard, (before, after)
'''

    env = os.environ.copy()
    # Preserve the test runner's build-tree package resolution in the child.
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    if completed.returncode != 0 and "SKIP_NO_RAISE_RANGE" in completed.stderr + completed.stdout:
        pytest.skip("RLIMIT_NOFILE hard limit leaves no lower soft-limit value to test")

    assert completed.returncode == 0, (
        "Thor import did not raise RLIMIT_NOFILE soft limit to the hard limit.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
