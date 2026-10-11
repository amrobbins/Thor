"""Focused Python logical RaggedTensor API contract (independent of GPU placement)."""

import pytest
import thor


def test_explicit_logical_m_and_shared_partition_token():
    token = thor.Tensor([1], thor.DataType.uint32)
    values = thor.Tensor([4, 3], thor.DataType.fp32)
    other_values = thor.Tensor([4, 2], thor.DataType.fp16)
    first = thor.RaggedTensor(values, row_partition_token=token, max_values_per_row=4)
    second = thor.RaggedTensor(other_values, row_partition_token=token, max_values_per_row=4)

    assert first.values == values
    assert first.values.get_dimensions() == [4, 3]
    assert first.row_partition_token == token
    assert first.offsets == token  # old name is a structural-token alias, not device offsets
    assert first.get_row_partition_token() == token
    assert first.max_values_per_row == 4
    assert first.shares_partition_with(second)
    assert first.row_partition_id == second.row_partition_id
    for obsolete in ("batch_size", "max_total_values", "get_batch_size", "get_max_total_values"):
        assert not hasattr(first, obsolete)


def test_logical_ragged_requires_semantic_m_and_token_shape():
    values = thor.Tensor([4, 3], thor.DataType.fp32)
    token = thor.Tensor([1], thor.DataType.uint64)
    with pytest.raises(TypeError):
        thor.RaggedTensor(values, token)
    with pytest.raises(TypeError):
        thor.RaggedTensor(thor.DataType.fp32, [3], 2, 8, 4)
    with pytest.raises(ValueError, match="max_values_per_row"):
        thor.RaggedTensor(values, token, 0)
    with pytest.raises(ValueError, match="leading logical dimension"):
        thor.RaggedTensor(values, token, 5)
    with pytest.raises(ValueError, match=r"shape \[1\]"):
        thor.RaggedTensor(values, thor.Tensor([3], thor.DataType.uint32), 4)
    with pytest.raises(ValueError, match="uint32 or uint64"):
        thor.RaggedTensor(values, thor.Tensor([1], thor.DataType.fp32), 4)
