import copy
import pytest
import thor


@pytest.mark.cuda
def test_stream_construct_from_gpu_num_and_introspection():
    s = thor.physical.Stream(gpu_num=0)
    assert s.get_gpu_num() == 0

    sid = s.get_id()
    assert isinstance(sid, int)
    assert sid >= 0

    # repr should be a string
    r = repr(s)
    assert isinstance(r, str)
    assert "Stream" in r


@pytest.mark.cuda
def test_stream_construct_from_placement():
    placement = thor.physical.Placement(thor.physical.DeviceType.gpu, 0)

    s = thor.physical.Stream(placement)
    assert s.get_gpu_num() == 0


@pytest.mark.cuda
def test_stream_put_event_wait_event_and_synchronize():
    s = thor.physical.Stream(gpu_num=0)

    e = s.put_event(enable_timing=False, expecting_host_to_wait=False)
    assert isinstance(e, thor.physical.Event)
    assert e.get_gpu_num() == 0

    # Make the stream wait on its own event (should be fine / no-op-ish)
    s.wait_event(e)

    # Synchronize both
    e.synchronize()
    s.synchronize()

    # Device-wide sync should also work
    thor.physical.Stream.device_synchronize(0)


@pytest.mark.cuda
def test_stream_copy_and_deepcopy():
    s1 = thor.physical.Stream(gpu_num=0)

    s2 = copy.copy(s1)
    assert isinstance(s2, thor.physical.Stream)
    assert s2.get_gpu_num() == 0

    s3 = copy.deepcopy(s1)
    assert isinstance(s3, thor.physical.Stream)
    assert s3.get_gpu_num() == 0
