import thor


def test_fixed_shape_parameter_create_storage_uses_input_placement():
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    input_desc = thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp16, [8, 16])
    input_tensor = thor.physical.PhysicalTensor(placement, input_desc)

    parameter = thor.Parameter(
        name="weights",
        shape=[16, 32],
        dtype=thor.DataType.fp32,
        trainable=True,
    )

    storage = parameter.create_storage(input_tensor)

    assert storage.get_placement() == placement
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp32
    assert storage.get_descriptor().get_dimensions() == [16, 32]


def test_python_parameter_subclass_can_create_custom_storage():

    class BiasLikeParameter(thor.Parameter):

        def __init__(self):
            super().__init__(name="biases", shape=[1], dtype=thor.DataType.fp16, trainable=True)

        def create_storage(self, input_tensor):
            output_features = input_tensor.get_descriptor().get_dimensions()[-1]
            return thor.physical.PhysicalTensor(
                input_tensor.get_placement(),
                thor.physical.PhysicalTensor.Descriptor(
                    input_tensor.get_descriptor().get_data_type(), [output_features]),
            )

    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    input_tensor = thor.physical.PhysicalTensor(
        placement,
        thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp16, [4, 7]),
    )

    storage = BiasLikeParameter().create_storage(input_tensor)

    assert storage.get_placement() == placement
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp16
    assert storage.get_descriptor().get_dimensions() == [7]


def test_python_parameter_subclass_can_delegate_to_default_create_storage_helper():

    class DynamicBiasParameter(thor.Parameter):

        def __init__(self):
            super().__init__(name="biases", shape=[1], dtype=thor.DataType.fp32, trainable=True)

        def create_storage(self, input_tensor):
            dims = input_tensor.get_descriptor().get_dimensions()
            shape = [dims[-1]]
            dtype = input_tensor.get_descriptor().get_data_type()
            return self.createStorage(input_tensor, shape=shape, dtype=dtype)

    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    input_tensor = thor.physical.PhysicalTensor(
        placement,
        thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp16, [4, 9]),
    )

    storage = DynamicBiasParameter().create_storage(input_tensor)

    assert storage.get_placement() == placement
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp16
    assert storage.get_descriptor().get_dimensions() == [9]
