import subprocess
import pytest
import thor


def _nvidia_smi_gpu_count() -> int:
    # Under @pytest.mark.cuda we expect nvidia-smi to exist and be functional
    r = subprocess.run(
        ["nvidia-smi", "-L"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3,
        check=True,
    )
    # Each GPU line typically begins with "GPU 0:" etc.
    return sum(1 for line in (r.stdout or "").splitlines() if line.strip().startswith("GPU "))


@pytest.mark.cuda
def test_machine_evaluator_minimal():
    me1 = thor.physical.MachineEvaluator.instance()
    me2 = thor.physical.MachineEvaluator.instance()
    assert me1 is me2  # singleton identity

    num = me1.get_num_gpus()
    assert isinstance(num, int)
    assert num >= 1

    # Cross-check against nvidia-smi (best-effort sanity check)
    smi_num = _nvidia_smi_gpu_count()
    assert num == smi_num

    ordered = me1.get_ordered_gpus()
    assert isinstance(ordered, (list, tuple))
    assert len(ordered) == num
    assert sorted(ordered) == list(range(num))

    for g in ordered:
        assert me1.get_gpu_type(g)
        assert me1.get_total_global_mem_bytes(g) > 0

    cur = me1.get_current_gpu_num()
    assert isinstance(cur, int)
    assert 0 <= cur < num

    # Low-hanging fruit: swap to GPU 0 and restore (should not crash)
    prev = thor.physical.MachineEvaluator.swap_active_device(0)
    assert isinstance(prev, int)
    assert 0 <= prev < num
    thor.physical.MachineEvaluator.swap_active_device(prev)
