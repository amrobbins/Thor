import copy

import pytest
import thor


@pytest.mark.cuda
def test_event_basic_record_and_synchronize():
    # Construct a stream on GPU 0 (priority forced to REGULAR in your binding)
    s = thor.physical.Stream(gpu_num=0)
    assert s.get_gpu_num() == 0

    # Construct an event and record it on the stream
    e = thor.physical.Event(gpu_num=0, enable_timing=False)
    assert e.get_gpu_num() == 0
    assert isinstance(e.get_id(), int)

    e.record(s)
    e.synchronize()  # should not throw / abort


@pytest.mark.cuda
def test_event_put_event_wait_event_and_copy():
    s = thor.physical.Stream(gpu_num=0)

    # put_event returns an Event already recorded on the stream
    e1 = s.put_event(enable_timing=False)
    assert isinstance(e1, thor.physical.Event)
    e1.synchronize()

    # wait_event should accept an Event
    s.wait_event(e1)
    s.synchronize()

    # copy / deepcopy
    e2 = copy.copy(e1)
    assert isinstance(e2, thor.physical.Event)

    e3 = copy.deepcopy(e1)
    assert isinstance(e3, thor.physical.Event)


@pytest.mark.cuda
def test_event_timing_elapsed_ms_non_negative():
    s = thor.physical.Stream(gpu_num=0)

    # Timing requires enable_timing=True on the events involved.
    start = s.put_event(enable_timing=True)
    # Ensure some ordering; synchronize stream so the start is definitely recorded
    s.synchronize()

    end = s.put_event(enable_timing=True)

    ms = end.synchronize_and_report_elapsed_time_ms(start)
    assert isinstance(ms, float)
    assert ms >= 0.0
