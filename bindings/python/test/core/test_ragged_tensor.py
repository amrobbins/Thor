"""Public logical ragged tensor behavior; physical buffers live in thor.physical."""

import pytest
import thor


def test_ragged_tensor_wraps_logical_values_and_canonical_partition_token():
    values = thor.Tensor([4, 4], thor.DataType.fp16)
    token = thor.Tensor([1], thor.DataType.uint64)

    ragged = thor.RaggedTensor(values, token, max_values_per_row=4)

    assert ragged.values == values
    assert ragged.offsets == token
    assert ragged.row_partition_token == token
    assert ragged.values_data_type == thor.DataType.fp16
    assert ragged.offsets_data_type == thor.DataType.uint64
    assert ragged.trailing_dimensions == [4]
    assert ragged.max_values_per_row == 4
    assert ragged.ragged_rank == 1
    assert not hasattr(ragged, "batch_size")
    assert not hasattr(ragged, "max_total_values")


def test_ragged_tensor_explicit_m_supports_multidimensional_trailing_values():
    ragged = thor.RaggedTensor(
        thor.Tensor([5, 2, 3], thor.DataType.bf16),
        thor.Tensor([1], thor.DataType.uint32),
        max_values_per_row=5,
    )
    assert ragged.values.get_dimensions() == [5, 2, 3]
    assert ragged.offsets.get_dimensions() == [1]
    assert ragged.trailing_dimensions == [2, 3]
    assert ragged.max_values_per_row == 5


def test_ragged_tensor_rejects_obsolete_physical_constructor_arguments():
    with pytest.raises(TypeError):
        thor.RaggedTensor(thor.DataType.fp32, [4], 5, 17, 6, thor.DataType.uint32)
    with pytest.raises(TypeError):
        thor.RaggedTensor(thor.Tensor([7, 4], thor.DataType.fp32), thor.Tensor([1], thor.DataType.uint32))


def test_ragged_tensor_values_require_exact_logical_m():
    values = thor.Tensor([12, 4], thor.DataType.fp16)
    token = thor.Tensor([1], thor.DataType.uint64)
    with pytest.raises(ValueError, match="leading logical dimension"):
        thor.RaggedTensor(values, token, max_values_per_row=7)


def test_ragged_tensor_row_partition_identity_is_shared_by_common_token():
    token = thor.Tensor([1], thor.DataType.uint32)
    first = thor.RaggedTensor(thor.Tensor([7, 2], thor.DataType.fp32), token, 7)
    second = thor.RaggedTensor(thor.Tensor([7, 5], thor.DataType.fp16), token, 7)
    different = thor.RaggedTensor(
        thor.Tensor([7, 2], thor.DataType.fp32),
        thor.Tensor([1], thor.DataType.uint32),
        7,
    )
    assert first.row_partition_id == second.row_partition_id
    assert first.shares_partition_with(second)
    assert not first.shares_partition_with(different)
