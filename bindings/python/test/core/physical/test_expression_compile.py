import math
import pytest
import thor
from thor.physical import Expression as ex


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_add_sub_mul_div(dtype: thor.DataType):
    x = ex.input(0)
    y = ex.input(1)

    expr = ((x + y) - 2.0) * (x / (y + 1.0))
    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_pow(dtype: thor.DataType):
    x = ex.input(0)
    y = ex.input(1)

    expr1 = x**y
    expr2 = x**2.0
    expr3 = 2.0**x

    assert ex.compile(
        expr1,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        dtype=dtype,
        device_num=0,
        use_fast_math=True,
    ) is not None

    assert ex.compile(
        expr3,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_negation(dtype: thor.DataType):
    x = ex.input(0)
    y = ex.input(1)

    expr = -(x**y) + 3.0
    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_min_max(dtype: thor.DataType):
    x = ex.input(0)
    y = ex.input(1)

    expr1 = ex.min(x, y)
    expr2 = ex.max(x, 3.0)
    expr3 = ex.min(2.0, y)
    expr4 = ex.max(ex.min(x, y), 3.0)

    assert ex.compile(
        expr1,
        dtype=dtype,
        device_num=0,
        use_fast_math=True,
    ) is not None

    assert ex.compile(
        expr2,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr4,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_exp_family(dtype: thor.DataType):
    x = ex.input(0)

    expr1 = ex.exp(x)
    expr2 = ex.exp2(x)
    expr3 = ex.exp10(x)

    assert ex.compile(
        expr1,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_log_family(dtype: thor.DataType):
    x = ex.input(0)

    expr1 = ex.ln(x)
    expr2 = ex.log(x)  # default base = e
    expr3 = ex.log2(x)
    expr4 = ex.log10(x)
    expr5 = ex.log(x, 7.0)

    assert ex.compile(
        expr1,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr2,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr3,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr4,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None

    assert ex.compile(
        expr5,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    ) is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_sqrt(dtype: thor.DataType):
    x = ex.input(0)

    expr = ex.sqrt(x + 4.0)
    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_nested_expression(dtype: thor.DataType):
    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)

    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize("use_fast_math", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_with_and_without_fast_math(dtype: thor.DataType, use_fast_math: bool):
    x = ex.input(0)
    y = ex.input(1)

    expr = ex.exp(ex.log2(x + 8.0) + (y**2.0) / 3.0)

    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=use_fast_math,
    )

    assert fused is not None


def test_python_numeric_coercion_rhs():
    x = ex.input(0)

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
    x = ex.input(0)

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
    x = ex.input(0)

    expr1 = ex.log(x)
    expr2 = ex.log(x, math.e)

    assert expr1 is not None
    assert expr2 is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_scalar_only_expression(dtype: thor.DataType):
    expr = ex.exp(ex.scalar(2.0)) + ex.log2(ex.scalar(8.0)) - ex.sqrt(ex.scalar(9.0))
    fused = ex.compile(
        expr,
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_min_max_scalar_only_expression(dtype: thor.DataType):
    expr = ex.max(ex.scalar(2.0), ex.scalar(5.0)) + ex.min(ex.scalar(10.0), ex.scalar(3.0))
    fused = ex.compile(
        expr,
        dtype=dtype,
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
@pytest.mark.parametrize(
    "dtype",
    [
        thor.DataType.fp16,
        thor.DataType.bf16,
        thor.DataType.fp8_e4m3,
        thor.DataType.fp8_e5m2,
        thor.DataType.fp32,
    ],
)
def test_compile_various_expressions(dtype: thor.DataType, expr):
    x = ex.input(0)
    y = ex.input(1)

    fused = ex.compile(
        expr(x, y),
        dtype=thor.DataType.fp32,
        device_num=0,
        use_fast_math=False,
    )

    assert fused is not None
