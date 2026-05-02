# bindings/python/test/core/physical/test_expression_flop_count.py

import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream


def _gpu_tensor(shape: list[int], dtype: thor.DataType = thor.DataType.fp32, gpu_num: int = 0) -> PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _stamp(eq, inputs: dict[str, PhysicalTensor]):
    stream = Stream(gpu_num=0)
    stamped = eq.stamp(inputs, stream)
    return stamped


def _assert_flops(
        eq, inputs: dict[str, PhysicalTensor], expected_total: int, expected_stage_flops: list[int] | None = None):
    stamped = _stamp(eq, inputs)

    assert stamped.flop_count() == expected_total

    stage_flops = stamped.stage_flop_counts()
    assert sum(stage_flops) == expected_total

    stage_kinds = eq._debug_stage_kinds(inputs)
    assert len(stage_flops) == len(stage_kinds)

    if expected_stage_flops is not None:
        assert stage_flops == expected_stage_flops

    return stage_kinds, stage_flops


@pytest.mark.cuda
def test_flop_count_single_fused_pointwise_stage():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    d = ex.input("d")

    expr = ((a + b) * c) - ex.sqrt(d)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "a": _gpu_tensor([2, 3, 4]),
        "b": _gpu_tensor([2, 3, 4]),
        "c": _gpu_tensor([2, 3, 4]),
        "d": _gpu_tensor([2, 3, 4]),
    }

    numel = 2 * 3 * 4
    expected = numel * 4  # add + mul + sqrt + sub

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=expected, expected_stage_flops=[expected])
    assert stage_kinds == ["FusedKernel"]


@pytest.mark.cuda
def test_flop_count_multi_output_shared_subexpression_counts_dag_semantics():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")

    shared = a + b
    outputs = ex.outputs({
        "sum": shared,
        "prod": shared * c,
    })
    eq = outputs.compile(device_num=0)

    inputs = {
        "a": _gpu_tensor([5, 7]),
        "b": _gpu_tensor([5, 7]),
        "c": _gpu_tensor([5, 7]),
    }

    numel = 5 * 7
    expected = numel * 2  # shared add counted once, then one mul for prod

    stage_kinds, stage_flops = _assert_flops(eq, inputs, expected_total=expected)

    # Current lowering may use one fused multi-output stage or multiple fused stages.
    assert all(kind == "FusedKernel" for kind in stage_kinds)
    assert sum(stage_flops) == expected


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("method_name", "expected_total", "expected_kind"),
    [
        ("reduce_sum", 16, "Reduction"),  # out_numel=8, reduce_extent=3 -> 8*(3-1)
        ("reduce_mean", 24, "Reduction"),  # out_numel=8, reduce_extent=3 -> 8*3
        ("reduce_norm1", 40, "Reduction"),  # out_numel=8, reduce_extent=3 -> 8*(2*3-1)
        ("reduce_norm2", 48, "Reduction"),  # out_numel=8, reduce_extent=3 -> 8*(2*3)
        ("argmax", 16, "ArgMinMax"),  # out_numel=8, reduce_extent=3 -> 8*(3-1)
    ],
)
def test_flop_count_reduction_and_arg_reduction_stages(method_name: str, expected_total: int, expected_kind: str):
    x = ex.input("x")
    expr = getattr(ex, method_name)(x, axis=[1], squeeze=False)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "x": _gpu_tensor([2, 3, 4]),
    }

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=expected_total, expected_stage_flops=[expected_total])
    assert stage_kinds == [expected_kind]


@pytest.mark.cuda
def test_flop_count_transpose_is_zero():
    x = ex.input("x")
    expr = x.transpose()
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "x": _gpu_tensor([5, 7]),
    }

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=0, expected_stage_flops=[0])
    assert stage_kinds == ["FusedKernel"]


@pytest.mark.cuda
def test_flop_count_matmul_uses_2mnk():
    a = ex.input("a")
    b = ex.input("b")

    expr = ex.matmul(a, b)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "a": _gpu_tensor([2, 3]),
        "b": _gpu_tensor([3, 4]),
    }

    expected = 2 * 2 * 4 * 3  # 2MNK

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=expected, expected_stage_flops=[expected])
    assert len(stage_kinds) == 1
    assert stage_kinds[0].startswith("Matmul")


@pytest.mark.cuda
def test_flop_count_gemm_includes_alpha_beta_scale_and_accumulate_terms():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")

    expr = ex.gemm(a, b, c, alpha=2.0, beta=3.0)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "a": _gpu_tensor([2, 3]),
        "b": _gpu_tensor([3, 4]),
        "c": _gpu_tensor([2, 4]),
    }

    out_numel = 2 * 4
    matmul_flops = 2 * 2 * 4 * 3  # 2MNK
    expected = matmul_flops + out_numel + out_numel + out_numel
    # alpha scale + beta scale + accumulation add

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=expected, expected_stage_flops=[expected])
    assert len(stage_kinds) == 1
    assert stage_kinds[0].startswith("Matmul")


@pytest.mark.cuda
def test_flop_count_convolution_forward():
    x = ex.input("x")
    w = ex.input("w")

    expr = ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "x": _gpu_tensor([2, 3, 5, 6], dtype=thor.DataType.fp16),  # N, C, H, W
        "w": _gpu_tensor([4, 3, 3, 2], dtype=thor.DataType.fp16),  # K, C, R, S
    }

    # Output dims:
    # N=2, K=4, OH=(5-3)+1=3, OW=(6-2)+1=5
    # FLOPs = 2 * N * K * OH * OW * C * R * S
    expected = 2 * 2 * 4 * 3 * 5 * 3 * 3 * 2

    stage_kinds, _ = _assert_flops(eq, inputs, expected_total=expected, expected_stage_flops=[expected])
    assert stage_kinds == ["Convolution"]


@pytest.mark.cuda
def test_flop_count_multistage_pointwise_then_reduction():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")

    expr = ex.reduce_sum((a + b) * c, axis=[1], squeeze=False)
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "a": _gpu_tensor([2, 3, 4]),
        "b": _gpu_tensor([2, 3, 4]),
        "c": _gpu_tensor([2, 3, 4]),
    }

    pointwise_flops = (2 * 3 * 4) * 2  # add + mul
    reduction_flops = (2 * 1 * 4) * (3 - 1)
    expected_total = pointwise_flops + reduction_flops

    stage_kinds, _ = _assert_flops(
        eq,
        inputs,
        expected_total=expected_total,
        expected_stage_flops=[pointwise_flops, reduction_flops],
    )
    assert stage_kinds == ["FusedKernel", "Reduction"]


@pytest.mark.cuda
def test_flop_count_reduce_max_backward_stage():
    x = ex.input("x")
    y = ex.reduce_max(x, axis=[1], squeeze=False)

    fwd_eq = ex.compile(y, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name="__grad_output")

    inputs = {
        "x": _gpu_tensor([2, 3, 4]),
        "__grad_output": _gpu_tensor([2, 1, 4]),
    }

    expected = (2 * 3 * 4) * 2

    stage_kinds, stage_flops = _assert_flops(bwd_eq, inputs, expected_total=expected)

    assert "ReduceMinMaxBackward" in stage_kinds
    assert sum(stage_flops) == expected


@pytest.mark.cuda
def test_flop_count_matmul_then_pointwise_stage():
    a = ex.input("a")
    b = ex.input("b")

    mm = ex.matmul(a, b)
    expr = mm * 2.0 + 1.0
    eq = ex.compile(expr, device_num=0)

    inputs = {
        "a": _gpu_tensor([2, 3]),
        "b": _gpu_tensor([3, 4]),
    }

    out_numel = 2 * 4
    matmul_flops = 2 * 2 * 4 * 3  # 2MNK
    pointwise_flops = out_numel * 2  # mul + add
    expected_total = matmul_flops + pointwise_flops

    stage_kinds, stage_flops = _assert_flops(
        eq,
        inputs,
        expected_total=expected_total,
        expected_stage_flops=[matmul_flops, pointwise_flops],
    )

    assert len(stage_kinds) == 2
    assert stage_kinds[0].startswith("Matmul")
    assert stage_kinds[1] == "FusedKernel"
