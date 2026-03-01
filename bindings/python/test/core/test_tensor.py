import pytest
import thor


def test_tensor_ctor_rejects_empty_dimensions():
    with pytest.raises(ValueError, match=r"dimensions cannot be size 0"):
        thor.Tensor([], thor.DataType.fp16)


@pytest.mark.parametrize(
    "dims, dtype",
    [
        ([1], thor.DataType.fp16),
        ([2, 3], thor.DataType.fp16),
        ([4, 5, 6], thor.DataType.fp32),
        ([7, 1, 9, 2], thor.DataType.int32),
    ],
)
def test_tensor_basic_properties_round_trip(dims, dtype):
    t = thor.Tensor(dims, dtype)

    # dimensions
    got_dims = t.get_dimensions()
    assert isinstance(got_dims, (list, tuple))
    assert list(got_dims) == dims

    # data type
    got_dtype = t.get_data_type()
    assert got_dtype == dtype

    # id
    tid = t.get_id()
    assert isinstance(tid, int)
    assert tid >= 0


@pytest.mark.parametrize(
    "dims, expected_elems",
    [
        ([1], 1),
        ([2, 3], 6),
        ([4, 5, 6], 120),
        ([7, 1, 9, 2], 126),
    ],
)
def test_tensor_total_num_elements(dims, expected_elems):
    t = thor.Tensor(dims, thor.DataType.fp16)
    assert t.get_total_num_elements() == expected_elems


@pytest.mark.parametrize(
    "dims, dtype",
    [
        ([2, 3], thor.DataType.fp16),
        ([2, 3], thor.DataType.fp32),
        ([2, 3], thor.DataType.int32),
        ([2, 3], thor.DataType.int8),
        ([4, 6], thor.DataType.packed_bool),
    ],
)
def test_tensor_bytes_per_element_and_total_size_in_bytes(dims, dtype):
    t = thor.Tensor(dims, dtype)

    bpe_static = thor.Tensor.bytes_per_element(dtype)
    bpe_instance = t.get_bytes_per_element()

    assert isinstance(bpe_static, float)
    assert isinstance(bpe_instance, float)
    assert bpe_static == bpe_instance
    assert bpe_static > 0

    expected_total_bytes = t.get_total_num_elements() * bpe_static
    assert t.get_total_size_in_bytes() == expected_total_bytes
