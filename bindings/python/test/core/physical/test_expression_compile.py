import math
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Placement, DeviceType

FLOAT_DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_add_sub_mul_div(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = ((x + y) - 2.0) * (x / (y + 1.0))
    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_pow(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr1 = x**y
    expr2 = x**2.0
    expr3 = 2.0**x

    assert ex.compile(
        expr1,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        device_num=0,
        use_fast_math=True,
    ) is not None

    assert ex.compile(
        expr3,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_negation(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = -(x**y) + 3.0
    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_min_max(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr1 = ex.min(x, y)
    expr2 = ex.max(x, 3.0)
    expr3 = ex.min(2.0, y)
    expr4 = ex.max(ex.min(x, y), 3.0)

    assert ex.compile(
        expr1,
        device_num=0,
        use_fast_math=True,
    ) is not None

    assert ex.compile(
        expr2,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr4,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_exp_family(dtype: thor.DataType):
    x = ex.input("x")

    expr1 = ex.exp(x)
    expr2 = ex.exp2(x)
    expr3 = ex.exp10(x)

    assert ex.compile(
        expr1,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_log_family(dtype: thor.DataType):
    x = ex.input("x")

    expr1 = ex.ln(x)
    expr2 = ex.log(x)  # default base = e
    expr3 = ex.log2(x)
    expr4 = ex.log10(x)
    expr5 = ex.log(x, 7.0)

    assert ex.compile(
        expr1,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr4,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr5,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_sqrt(dtype: thor.DataType):
    x = ex.input("x")

    expr = ex.sqrt(x + 4.0)
    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_nested_expression(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("use_fast_math", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_with_and_without_fast_math(dtype: thor.DataType, use_fast_math: bool):
    x = ex.input("x")
    y = ex.input("y")

    expr = ex.exp(ex.log2(x + 8.0) + (y**2.0) / 3.0)

    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=use_fast_math,
    )

    assert fused is not None


def test_python_numeric_coercion_rhs():
    x = ex.input("x")

    expr1 = x + 3.0
    expr2 = x - 2
    expr3 = x * 5.0
    expr4 = x / 7
    expr5 = x**3
    expr6 = ex.min(x, 2.0)
    expr7 = ex.max(x, 4)

    assert expr1 is not None
    assert expr2 is not None
    assert expr3 is not None
    assert expr4 is not None
    assert expr5 is not None
    assert expr6 is not None
    assert expr7 is not None


def test_python_numeric_coercion_lhs():
    x = ex.input("x")

    expr1 = 3.0 + x
    expr2 = 2 - x
    expr3 = 5.0 * x
    expr4 = 7 / x
    expr5 = 2**x
    expr6 = ex.min(2.0, x)
    expr7 = ex.max(4, x)

    assert expr1 is not None
    assert expr2 is not None
    assert expr3 is not None
    assert expr4 is not None
    assert expr5 is not None
    assert expr6 is not None
    assert expr7 is not None


def test_log_default_base_matches_explicit_e_construction():
    x = ex.input("x")

    expr1 = ex.log(x)
    expr2 = ex.log(x, math.e)

    assert expr1 is not None
    assert expr2 is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_scalar_only_expression(dtype: thor.DataType):
    expr = ex.exp(ex.scalar(2.0)) + ex.log2(ex.scalar(8.0)) - ex.sqrt(ex.scalar(9.0))
    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_min_max_scalar_only_expression(dtype: thor.DataType):
    expr = ex.max(ex.scalar(2.0), ex.scalar(5.0)) + ex.min(ex.scalar(10.0), ex.scalar(3.0))
    fused = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "expr",
    [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x**y,
        lambda x, y: ex.min(x, y),
        lambda x, y: ex.max(x, y),
        lambda x, y: ex.exp(x),
        lambda x, y: ex.exp2(x),
        lambda x, y: ex.exp10(x),
        lambda x, y: ex.ln(x + 5.0),
        lambda x, y: ex.log(x + 5.0),
        lambda x, y: ex.log2(x + 8.0),
        lambda x, y: ex.log10(x + 10.0),
        lambda x, y: ex.sqrt(x + 4.0),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_various_expressions(dtype: thor.DataType, expr):
    x = ex.input("x")
    y = ex.input("y")

    fused = ex.compile(
        expr(x, y),
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
def test_fused_equation_output_names_single_output():
    x = ex.input("x")
    y = ex.input("y")
    expr = (x + 1.0) * (y - 2.0)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    assert eq.output_names() == ["output"]


@pytest.mark.cuda
def test_fused_equation_output_names_multi_output():
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
        "shifted": x + 2.0,
    })

    eq = outs.compile(device_num=0, use_fast_math=False)

    assert eq.output_names() == ["sum", "prod", "shifted"]


@pytest.mark.cuda
def test_fused_equation_output_shape_single_input_single_output_from_tensor():
    x = ex.input("x")
    expr = (x + 1.0) * 2.0

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    shape = eq.output_shape(x_gpu)

    assert shape == [2, 3, 4]


@pytest.mark.cuda
def test_fused_equation_output_shape_single_input_single_output_from_dict():
    x = ex.input("x")
    expr = ex.sqrt(x + 1.0)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([5, 7], thor.DataType.fp32)
    shape = eq.output_shape({
        "x": x_gpu
    })

    assert shape == [5, 7]


@pytest.mark.cuda
def test_fused_equation_output_shapes_single_input_single_output_from_tensor():
    x = ex.input("x")
    expr = ex.ln(x + 2.0)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([4, 6], thor.DataType.fp32)
    shapes = eq.output_shapes(x_gpu)

    assert shapes == {
        "output": [4, 6]
    }


@pytest.mark.cuda
def test_fused_equation_output_shapes_single_input_single_output_from_dict():
    x = ex.input("x")
    expr = x * 3.0

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([8, 2], thor.DataType.fp32)
    shapes = eq.output_shapes({
        "x": x_gpu
    })

    assert shapes == {
        "output": [8, 2]
    }


@pytest.mark.cuda
def test_fused_equation_output_shape_single_output_broadcast_from_dict():
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 1, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3, 4], thor.DataType.fp32)

    shape = eq.output_shape({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shape == [2, 3, 4]


@pytest.mark.cuda
def test_fused_equation_output_shapes_multi_output_same_domain():
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 1, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3, 4], thor.DataType.fp32)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shapes == {
        "sum": [2, 3, 4],
        "prod": [2, 3, 4],
    }


@pytest.mark.cuda
def test_fused_equation_output_shapes_multi_output_different_domains():
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    outs = ex.outputs({
        "xy_sum": x + y,  # [2, 3, 4]
        "xz_mul": x * z,  # [2, 1, 4]
        "y_shift": y + 1.0,  # [1, 3, 4]
    })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 1, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3, 4], thor.DataType.fp32)
    z_gpu = _gpu_tensor([2, 1, 4], thor.DataType.fp32)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
        "z": z_gpu,
    })

    assert shapes == {
        "xy_sum": [2, 3, 4],
        "xz_mul": [2, 1, 4],
        "y_shift": [1, 3, 4],
    }


@pytest.mark.cuda
def test_fused_equation_output_shape_reduction_single_output():
    x = ex.input("x")
    expr = ex.reduce_sum(x + 1.0, axis=1, squeeze=False)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    shape = eq.output_shape(x_gpu)

    assert shape == [2, 1, 4]


@pytest.mark.cuda
def test_fused_equation_output_shape_reduction_single_output_squeezed():
    x = ex.input("x")
    expr = ex.reduce_sum(x + 1.0, axis=[1, 2], squeeze=True)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    shape = eq.output_shape({
        "x": x_gpu
    })

    assert shape == [2]


@pytest.mark.cuda
def test_fused_equation_output_shapes_multi_output_with_reduction_and_epilogue():
    x = ex.input("x")
    y = ex.input("y")

    trunk = (x + 1.0) * (y - 0.5)

    outs = ex.outputs(
        {
            "reduced": ex.reduce_sum(trunk, axis=2, squeeze=False),  # [B, M, 1]
            "final": ex.sqrt(ex.reduce_sum(trunk, axis=2, squeeze=False) + 1.0),  # [B, M, 1]
            "pointwise": x + y,  # [B, M, N]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shapes == {
        "reduced": [2, 3, 1],
        "final": [2, 3, 1],
        "pointwise": [2, 3, 4],
    }


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


@pytest.mark.cuda
def test_fused_equation_output_shape_rejects_multi_output_equation():
    x = ex.input("x")

    outs = ex.outputs({
        "a": x + 1.0,
        "b": x * 2.0,
    })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)
    x_gpu = _gpu_tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="multiple final outputs|getOutputShape"):
        eq.output_shape(x_gpu)


@pytest.mark.cuda
def test_fused_equation_output_shape_rejects_single_tensor_for_multi_input_equation():
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    x_gpu = _gpu_tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="single input|requires 2 inputs|getOutputShapes"):
        eq.output_shape(x_gpu)


@pytest.mark.cuda
def test_fused_equation_output_shapes_rejects_unexpected_input_name():
    x = ex.input("x")
    expr = x + 1.0

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    x_gpu = _gpu_tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="Unexpected input"):
        eq.output_shapes({
            "x": x_gpu,
            "wrong_name": x_gpu
        })


@pytest.mark.cuda
def test_fused_equation_output_shapes_rejects_missing_required_input():
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    x_gpu = _gpu_tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="Missing required fused equation input"):
        eq.output_shapes({
            "x": x_gpu
        })


@pytest.mark.cuda
def test_fused_equation_output_shape_as_type_single_input_pointwise():
    x = ex.input("x", as_type=thor.DataType.fp16)
    expr = (x + 1.0) * 2.0

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17], thor.DataType.fp32)
    shape = eq.output_shape({
        "x": x_gpu
    })

    assert shape == [17]


@pytest.mark.cuda
def test_fused_equation_output_shape_as_type_single_output_broadcast_promoted():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")
    expr = x + y

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17, 1], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3], thor.DataType.fp32)

    shape = eq.output_shape({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shape == [17, 3]


@pytest.mark.cuda
def test_fused_equation_output_shape_as_type_direct_reduction_single_output():
    x = ex.input("x", as_type=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=2, squeeze=False)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    shape = eq.output_shape({
        "x": x_gpu
    })

    assert shape == [2, 3, 1]


@pytest.mark.cuda
def test_fused_equation_output_shape_as_type_direct_reduction_single_output_squeezed():
    x = ex.input("x", as_type=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=[1, 2], squeeze=True)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    shape = eq.output_shape({
        "x": x_gpu
    })

    assert shape == [2]


@pytest.mark.cuda
def test_fused_equation_output_shapes_as_type_multi_output_with_broadcast_and_mixed_shapes():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")

    xy = x + y
    y_shift = y - 0.5

    outs = ex.outputs(
        {
            "wide_sum": xy,  # [17, 3]
            "wide_mix": xy * y_shift,  # [17, 3]
            "narrow_shift": x + 1.0,  # [17, 1]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17, 1], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3], thor.DataType.fp32)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shapes == {
        "wide_sum": [17, 3],
        "wide_mix": [17, 3],
        "narrow_shift": [17, 1],
    }


@pytest.mark.cuda
def test_fused_equation_output_shapes_as_type_multi_output_with_reduction_and_epilogue():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")

    trunk = (x + 1.0) * (y - 0.5)

    outs = ex.outputs(
        {
            "reduced": ex.reduce_sum(trunk, axis=2, squeeze=False),  # [B, M, 1]
            "final": ex.sqrt(ex.reduce_sum(trunk, axis=2, squeeze=False) + 1.0),  # [B, M, 1]
            "pointwise": x + 2.0,  # [B, M, N]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shapes == {
        "reduced": [2, 3, 1],
        "final": [2, 3, 1],
        "pointwise": [2, 3, 4],
    }


@pytest.mark.cuda
def test_fused_equation_output_shapes_as_type_homogeneous_broadcast_same_runtime_dtype():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17, 1], thor.DataType.fp16)
    y_gpu = _gpu_tensor([1, 3], thor.DataType.fp16)

    shapes = eq.output_shapes({
        "x": x_gpu,
        "y": y_gpu,
    })

    assert shapes == {
        "sum": [17, 3],
        "prod": [17, 3],
    }
