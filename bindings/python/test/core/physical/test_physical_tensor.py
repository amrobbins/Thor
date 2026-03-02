import copy
import pytest
import thor


def test_physical_tensor_nested_types_exist():
    assert hasattr(thor, "physical")
    assert hasattr(thor.physical, "PhysicalTensor")

    PT = thor.physical.PhysicalTensor
    assert hasattr(PT, "Descriptor")

    # These should also exist as top-level types in thor.physical
    assert hasattr(thor.physical, "Placement")
    assert hasattr(thor.physical, "DeviceType")
    assert not hasattr(thor.physical, "Descriptor")


def test_placement_basic_round_trip_and_eq():
    PT = thor.physical.PhysicalTensor
    DT = thor.physical.DeviceType

    p0 = thor.physical.Placement(DT.cpu, 0)
    assert p0.get_device_type() == DT.cpu
    assert p0.get_device_num() == 0
    assert "CPU" in str(p0)

    p1 = thor.physical.Placement(DT.cpu, 0)
    p2 = thor.physical.Placement(DT.gpu, 0)

    assert p0 == p1
    assert p0 != p2


def test_descriptor_ctor_rejects_empty_dims():
    with pytest.raises(ValueError, match="dimensions cannot be empty"):
        thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp16, [])


def test_descriptor_basic_properties_and_helpers():
    D = thor.physical.PhysicalTensor.Descriptor

    desc = D(thor.DataType.fp16, [2, 3, 4])
    assert desc.get_data_type() == thor.DataType.fp16
    assert list(desc.get_dimensions()) == [2, 3, 4]
    assert desc.get_num_dimensions() == 3
    assert desc.get_total_num_elements() == 24

    s = str(desc)
    r = repr(desc)
    assert isinstance(s, str) and len(s) > 0
    assert isinstance(r, str) and "TensorDescriptor" in r

    # element name/type name
    assert desc.get_element_type_name() == "fp16"
    assert D.element_type_name(thor.DataType.fp16) == "fp16"

    # element size
    assert D.element_size_in_bytes(thor.DataType.fp16) == 2
    assert D.element_size_in_bytes(thor.DataType.packed_bool) == 0.125
    assert D.element_size_in_bytes(thor.DataType.uint64) == 8

    # size math
    assert desc.get_array_size_in_bytes() == 24 * 2
    assert D.array_size_in_bytes(24, thor.DataType.fp16) == 48

    with pytest.raises(ValueError, match="num_elements must be"):
        D.array_size_in_bytes(-1, thor.DataType.fp16)

    # predicates
    assert desc.is_integral_type() is False
    assert D.is_integral_data_type(thor.DataType.int32) is True

    assert desc.is_boolean_type() is False
    assert D.is_boolean_data_type(thor.DataType.bool) is True

    assert desc.is_signed_type() is True
    assert D.is_signed_data_type(thor.DataType.uint32) is False


def test_descriptor_reshape_and_indexing_helpers():
    D = thor.physical.PhysicalTensor.Descriptor

    desc = D(thor.DataType.fp16, [2, 3, 4])
    assert desc.dimension_stride(0) == 12
    assert desc.dimension_stride(1) == 4
    assert desc.dimension_stride(2) == 1

    # flat/dimensional index round-trip
    flat = desc.flat_index([1, 2, 3])  # 1*12 + 2*4 + 3 = 23
    assert flat == 23
    assert list(desc.dimensional_index(flat)) == [1, 2, 3]

    # reshape with same total elements
    desc.reshape([4, 6])  # 24 elems
    assert desc.get_total_num_elements() == 24
    assert list(desc.get_dimensions()) == [4, 6]

    with pytest.raises(ValueError, match="new_dimensions cannot be empty"):
        desc.reshape([])


def test_physical_tensor_construct_copy_deepcopy_cpu():
    PT = thor.physical.PhysicalTensor
    DT = thor.physical.DeviceType

    placement = thor.physical.Placement(DT.cpu, 0)
    desc = PT.Descriptor(thor.DataType.fp16, [8, 8])

    t = PT(placement, desc)  # alignment_bytes default (256)
    assert t is not None
    assert "<thor.physical.PhysicalTensor" in repr(t)
    assert t.get_descriptor() == desc
    assert t.get_placement() == placement
    assert t.get_size_in_bytes() == desc.get_array_size_in_bytes() == 2 * (8 * 8)

    t2 = copy.copy(t)
    assert t2 is not None
    assert type(t2) is type(t)

    t3 = copy.deepcopy(t)
    assert t3 is not None
    assert type(t3) is type(t)


def test_physical_tensor_nested_aliases_match_top_level_types():
    PT = thor.physical.PhysicalTensor

    assert thor.physical.Placement is thor.physical.Placement
    assert thor.physical.DeviceType is thor.physical.DeviceType
    assert PT.Descriptor is thor.physical.PhysicalTensor.Descriptor


def test_physical_tensor_get_descriptor_get_placement_and_size_in_bytes():
    PT = thor.physical.PhysicalTensor
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    desc = PT.Descriptor(thor.DataType.fp16, [2, 3, 4])

    t = PT(placement, desc)

    got_desc = t.get_descriptor()
    assert isinstance(got_desc, PT.Descriptor)
    assert got_desc == desc

    got_place = t.get_placement()
    assert isinstance(got_place, thor.physical.Placement)
    assert got_place == placement

    size = t.get_size_in_bytes()
    assert isinstance(size, int)
    assert size == desc.get_array_size_in_bytes()
    assert size == 2 * (2 * 3 * 4)
