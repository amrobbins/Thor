import pytest
import thor


def _net():
    return thor.Network("test_net_split")


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def test_split_gives_pieces_of_their_sizes_in_the_input_type():
    n = _net()
    layer = thor.layers.Split(n, _tensor([3, 7, 5]), 1, [2, 4, 1])
    assert isinstance(layer, thor.layers.Split)
    pieces = layer.get_feature_outputs()
    assert [piece.get_dimensions() for piece in pieces] == [[3, 2, 5], [3, 4, 5], [3, 1, 5]]
    assert all(piece.get_data_type() == thor.DataType.fp32 for piece in pieces)
    assert layer.axis == 1
    assert layer.sizes == [2, 4, 1]


def test_split_converts_each_piece_to_its_output_data_type():
    n = _net()
    layer = thor.layers.Split(
        n, _tensor([4, 6]), 1, [1, 5], output_data_types=[thor.DataType.int32, thor.DataType.bf16]
    )
    pieces = layer.get_feature_outputs()
    assert pieces[0].get_data_type() == thor.DataType.int32
    assert pieces[1].get_data_type() == thor.DataType.bf16
    assert pieces[1].get_dimensions() == [4, 5]


@pytest.mark.parametrize(
    "axis, sizes, types, message",
    [
        (1, [2, 4], None, r"sum to 6"),
        (1, [7, 0], None, r"positive"),
        (1, [7], None, r"two or more"),
        (3, [2, 3], None, r"out of range"),
        (1, [3, 4], [thor.DataType.fp32], r"one output data type per piece"),
    ],
)
def test_split_rejects_sizes_axis_and_types_it_cannot_take(axis, sizes, types, message):
    n = _net()
    with pytest.raises(ValueError, match=message):
        thor.layers.Split(n, _tensor([3, 7, 5]), axis, sizes, output_data_types=types)


def _cpu_tensor_from_numpy(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _ragged_split_network(name):
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [6],
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=4,
    )
    split = thor.layers.Split(
        network, history, 0, [2, 3, 1], output_data_types=[thor.DataType.fp32, thor.DataType.fp16, thor.DataType.fp32]
    )
    pieces = split.get_feature_outputs()
    for name_, piece in zip(["a", "b", "c"], pieces):
        thor.layers.RaggedNetworkOutput(network, name_, piece)
    return network, history, split, pieces


def test_split_ragged_pieces_share_the_input_partition():
    _, history, split, pieces = _ragged_split_network("ragged-split-shape")
    assert split.use_ragged
    assert all(isinstance(piece, thor.RaggedTensor) for piece in pieces)
    assert [piece.get_trailing_dimensions() for piece in pieces] == [[2], [3], [1]]
    assert all(piece.offsets == history.offsets for piece in pieces)
    assert pieces[1].values.get_data_type() == thor.DataType.fp16


def test_split_ragged_axis_counts_the_trailing_dimensions():
    network = thor.Network("ragged-split-axis")
    history = thor.layers.RaggedNetworkInput(
        network, "history", thor.DataType.fp32, [6], offsets_data_type=thor.DataType.uint32, max_values_per_row=4
    )
    with pytest.raises(ValueError, match=r"out of range"):
        thor.layers.Split(network, history, 1, [3, 3])


@pytest.mark.cuda
def test_split_ragged_forward_and_save_load_move_only_the_active_prefix(tmp_path):
    import numpy as np

    name = "ragged-split-forward-save-load"
    network, _, _, _ = _ragged_split_network(name)
    offsets = np.asarray([0, 3, 3, 7], dtype=np.uint32)
    values = (np.arange(3 * 4 * 6, dtype=np.float32).reshape(3 * 4, 6) * 0.5).astype(np.float32)
    # The inactive capacity is poisoned; Split must not read it.
    values[7:] = np.nan
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor_from_numpy(values, thor.DataType.fp32),
        _cpu_tensor_from_numpy(offsets, thor.DataType.uint32),
        max_values_per_row=4,
    )

    def check(placed):
        result = placed.infer({"history": physical})
        for name_, (first, last) in zip(["a", "b", "c"], [(0, 2), (2, 5), (5, 6)]):
            np.testing.assert_array_equal(result[name_].offsets.numpy(), offsets)
            observed = np.asarray(result[name_].values.numpy())[:7].astype(np.float32)
            np.testing.assert_array_equal(observed, values[:7, first:last])

    check(network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1))
    save_dir = tmp_path / "ragged_split"
    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    check(loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1))
