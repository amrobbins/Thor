import thor


def _cpu_input(dtype=thor.DataType.fp16, dims=(8, 16)):
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    desc = thor.physical.PhysicalTensor.Descriptor(dtype, list(dims))
    return thor.physical.PhysicalTensor(placement, desc)


def test_fixed_shape_parameter_create_storage_uses_input_placement():
    input_tensor = _cpu_input()

    parameter = thor.Parameter(
        name="weights",
        shape=[16, 32],
        dtype=thor.DataType.fp32,
        trainable=True,
    )

    storage = parameter.create_storage(input_tensor)

    assert storage.get_placement() == input_tensor.get_placement()
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp32
    assert storage.get_descriptor().get_dimensions() == [16, 32]


def test_storage_context_exposes_named_inputs():
    x = _cpu_input(dims=(4, 7))
    y = _cpu_input(dims=(4, 9))
    ctx = thor.Parameter.StorageContext({
        "x": x,
        "y": y
    })

    assert ctx.has_input("x")
    assert ctx.has_input("y")
    assert ctx.input_names() == ["x", "y"]
    assert ctx.get_input("x").get_descriptor().get_dimensions() == [4, 7]


def test_python_parameter_subclass_can_create_custom_storage_from_context():

    class BiasLikeParameter(thor.Parameter):

        def __init__(self):
            super().__init__(name="biases", shape=[1], dtype=thor.DataType.fp16, trainable=True)

        def create_storage(self, ctx):
            input_tensor = ctx.get_input("feature_input")
            output_features = input_tensor.get_descriptor().get_dimensions()[-1]
            return thor.physical.PhysicalTensor(
                input_tensor.get_placement(),
                thor.physical.PhysicalTensor.Descriptor(
                    input_tensor.get_descriptor().get_data_type(), [output_features]),
            )

    input_tensor = _cpu_input(dims=(4, 7))
    ctx = thor.Parameter.StorageContext(input_tensor)

    storage = BiasLikeParameter().create_storage(ctx)

    assert storage.get_placement() == input_tensor.get_placement()
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp16
    assert storage.get_descriptor().get_dimensions() == [7]


def test_python_parameter_subclass_can_delegate_to_default_create_storage_helper():

    class DynamicBiasParameter(thor.Parameter):

        def __init__(self):
            super().__init__(name="biases", shape=[1], dtype=thor.DataType.fp32, trainable=True)

        def create_storage(self, ctx):
            input_tensor = ctx.get_input("feature_input")
            dims = input_tensor.get_descriptor().get_dimensions()
            shape = [dims[-1]]
            dtype = input_tensor.get_descriptor().get_data_type()
            return self.createStorage(input_tensor, shape=shape, dtype=dtype)

    input_tensor = _cpu_input(dims=(4, 9))
    ctx = thor.Parameter.StorageContext(input_tensor)

    storage = DynamicBiasParameter().create_storage(ctx)

    assert storage.get_placement() == input_tensor.get_placement()
    assert storage.get_descriptor().get_data_type() == thor.DataType.fp16
    assert storage.get_descriptor().get_dimensions() == [9]
