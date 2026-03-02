import pytest
import thor


@pytest.mark.cuda
def test_scoped_gpu_restores_previous_device():
    me = thor.physical.MachineEvaluator.instance()
    num = me.get_num_gpus()
    assert num >= 1

    original = me.get_current_gpu_num()
    assert 0 <= original < num

    # Pick a target device. If only one GPU, target = original (still should be safe/no-op).
    target = 0 if original != 0 else (1 if num > 1 else 0)

    with thor.physical.ScopedGpu(target):
        cur = me.get_current_gpu_num()
        assert cur == target

    # After context exits, device should be restored
    restored = me.get_current_gpu_num()
    assert restored == original


@pytest.mark.cuda
def test_scoped_gpu_nested_contexts_restore_correctly():
    me = thor.physical.MachineEvaluator.instance()
    num = me.get_num_gpus()
    assert num >= 1

    original = me.get_current_gpu_num()

    # Choose up to two distinct targets if possible
    t1 = 0 if original != 0 else (1 if num > 1 else 0)
    t2 = (t1 + 1) % num if num > 1 else t1

    with thor.physical.ScopedGpu(t1):
        assert me.get_current_gpu_num() == t1
        with thor.physical.ScopedGpu(t2):
            assert me.get_current_gpu_num() == t2
        # inner should restore back to t1
        assert me.get_current_gpu_num() == t1

    # outer should restore back to original
    assert me.get_current_gpu_num() == original
