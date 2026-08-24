from __future__ import annotations

import thor


def test_ragged_tensor_wraps_packed_values_and_canonical_offsets():
    values = thor.Tensor([12, 4], thor.DataType.fp16)
    offsets = thor.Tensor([4], thor.DataType.uint64)

    ragged = thor.RaggedTensor(values, offsets)

    assert ragged.values == values
    assert ragged.offsets == offsets
    assert ragged.values_data_type == thor.DataType.fp16
    assert ragged.offsets_data_type == thor.DataType.uint64
    assert ragged.trailing_dimensions == [4]
    assert ragged.batch_size == 3
    assert ragged.max_total_values == 12
    assert ragged.max_values_per_row is None
    assert ragged.ragged_rank == 1


def test_ragged_tensor_descriptor_constructor_uses_canonical_layout():
    ragged = thor.RaggedTensor(
        thor.DataType.bf16,
        [2, 3],
        5,
        17,
        thor.DataType.uint32,
    )

    assert ragged.values.get_dimensions() == [17, 2, 3]
    assert ragged.offsets.get_dimensions() == [6]
    assert ragged.trailing_dimensions == [2, 3]
    assert ragged.batch_size == 5
    assert ragged.max_total_values == 17


def test_ragged_tensor_descriptor_constructor_accepts_max_values_per_row():
    ragged = thor.RaggedTensor(
        thor.DataType.fp32,
        [4],
        5,
        17,
        6,
        thor.DataType.uint32,
    )

    assert ragged.batch_size == 5
    assert ragged.max_total_values == 17
    assert ragged.max_values_per_row == 6


def test_ragged_tensor_wrapped_values_can_declare_max_values_per_row():
    values = thor.Tensor([12, 4], thor.DataType.fp16)
    offsets = thor.Tensor([4], thor.DataType.uint64)

    ragged = thor.RaggedTensor(values, offsets, 7)

    assert ragged.max_values_per_row == 7
    assert ragged.max_total_values == 12
