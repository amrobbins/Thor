from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import math
import os
from dataclasses import dataclass, replace
from math import prod
from typing import Callable, Iterator

import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream

GPU_NUM = int(os.getenv("THOR_EMBEDDING_PERF_GPU", os.getenv("THOR_EXPR_PERF_GPU", "0")))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


class _NvtxHooks:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lib = None
        self._available = False
        self._warned = False
        if not enabled:
            return

        candidates: list[str] = []
        found = ctypes.util.find_library("nvToolsExt")
        if found:
            candidates.append(found)
        candidates.extend(["libnvToolsExt.so.1", "libnvToolsExt.so"])
        for candidate in candidates:
            try:
                self._lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if self._lib is None:
            return

        self._lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._lib.nvtxRangePushA.restype = ctypes.c_int
        self._lib.nvtxRangePop.argtypes = []
        self._lib.nvtxRangePop.restype = ctypes.c_int
        self._available = True

    @property
    def active(self) -> bool:
        return self.enabled and self._available

    def _warn_unavailable_once(self) -> None:
        if self.enabled and not self._available and not self._warned:
            print("THOR_EMBEDDING_PERF_NVTX=1 requested, but libnvToolsExt could not be loaded; NVTX ranges disabled.")
            self._warned = True

    @contextlib.contextmanager
    def range(self, name: str) -> Iterator[None]:
        if not self.active:
            self._warn_unavailable_once()
            yield
            return
        assert self._lib is not None
        self._lib.nvtxRangePushA(name.encode("utf-8"))
        try:
            yield
        finally:
            self._lib.nvtxRangePop()


class _CudaProfilerHooks:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lib = None
        self._available = False
        self._warned = False
        if not enabled:
            return

        candidates: list[str] = []
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend([
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
        ])
        for candidate in candidates:
            try:
                self._lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if self._lib is None:
            return

        self._lib.cudaProfilerStart.argtypes = []
        self._lib.cudaProfilerStart.restype = ctypes.c_int
        self._lib.cudaProfilerStop.argtypes = []
        self._lib.cudaProfilerStop.restype = ctypes.c_int
        self._available = True

    @property
    def active(self) -> bool:
        return self.enabled and self._available

    def _warn_unavailable_once(self) -> None:
        if self.enabled and not self._available and not self._warned:
            print("THOR_EMBEDDING_PERF_CUDA_PROFILER=1 requested, but libcudart could not be loaded; profiler API disabled.")
            self._warned = True

    def start(self) -> None:
        if not self.active:
            self._warn_unavailable_once()
            return
        assert self._lib is not None
        status = int(self._lib.cudaProfilerStart())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStart failed with status {status}")

    def stop(self) -> None:
        if not self.active:
            return
        assert self._lib is not None
        status = int(self._lib.cudaProfilerStop())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {status}")


class _EmbeddingProfilerHooks:
    def __init__(self) -> None:
        self.nvtx = _NvtxHooks(_env_bool("THOR_EMBEDDING_PERF_NVTX", False))
        self.cuda_profiler = _CudaProfilerHooks(_env_bool("THOR_EMBEDDING_PERF_CUDA_PROFILER", False))
        self.nvtx_launch_ranges = _env_bool("THOR_EMBEDDING_PERF_NVTX_LAUNCH_RANGES", False)

    @contextlib.contextmanager
    def range(self, name: str) -> Iterator[None]:
        with self.nvtx.range(name):
            yield

    @contextlib.contextmanager
    def cuda_profiled_region(self) -> Iterator[None]:
        self.cuda_profiler.start()
        try:
            yield
        finally:
            self.cuda_profiler.stop()


PROFILE_HOOKS = _EmbeddingProfilerHooks()

# Keep defaults practical while still large enough to avoid measuring a repeatedly
# L2-resident gather.  The benchmark rotates across independent stamped launches
# whose index tensors select disjoint row windows from the embedding table.
WARMUP_ITERS = int(os.getenv("THOR_EMBEDDING_PERF_WARMUP_ITERS", "8"))
MEASURE_ITERS = int(os.getenv("THOR_EMBEDDING_PERF_MEASURE_ITERS", "64"))
MIN_ROTATING_POOL_BYTES = int(os.getenv("THOR_EMBEDDING_PERF_MIN_POOL_BYTES", str(512 * 1024 * 1024)))
MAX_POOL_SLOTS = int(os.getenv("THOR_EMBEDDING_PERF_MAX_POOL_SLOTS", "32"))
EXPLICIT_POOL_SLOTS = os.getenv("THOR_EMBEDDING_PERF_POOL_SLOTS")
CASE_FILTER = os.getenv("THOR_EMBEDDING_PERF_CASE_FILTER")
ENABLE_LARGE_CASES = os.getenv("THOR_EMBEDDING_PERF_ENABLE_LARGE_CASES", "0") != "0"
ROTATE_WEIGHTS = os.getenv("THOR_EMBEDDING_PERF_ROTATE_WEIGHTS", "0") != "0"

# Nsight Compute replays a single profiled kernel multiple times.  If that single
# launch only touches tens of MiB, the replay can become an L2 benchmark instead
# of a DRAM benchmark even though the normal pytest loop rotates across slots.
# Hard-DRAM mode scales token counts so each individual profiled launch reads at
# least this many table bytes, defaults the rotating pool to a small fixed size to
# keep memory use practical, and initializes weights with non-compressible data.
HARD_DRAM_PROFILE = _env_bool("THOR_EMBEDDING_PERF_HARD_DRAM", False)
HARD_DRAM_MIN_TABLE_READ_BYTES = int(
    os.getenv("THOR_EMBEDDING_PERF_HARD_DRAM_MIN_TABLE_READ_BYTES", str(256 * 1024 * 1024))
)
HARD_DRAM_POOL_SLOTS = int(os.getenv("THOR_EMBEDDING_PERF_HARD_DRAM_POOL_SLOTS", "2"))
EXPLICIT_NUM_TOKENS = os.getenv("THOR_EMBEDDING_PERF_NUM_TOKENS")

# Initializing the large table is not needed for a raw copy-throughput smoke
# benchmark.  For profiling, initialized random data is useful because it avoids
# reading uninitialized pages and avoids L2 compression artifacts from all-zero
# table/output values.
RANDOMIZE_WEIGHTS = _env_bool("THOR_EMBEDDING_PERF_RANDOMIZE_WEIGHTS", HARD_DRAM_PROFILE)
INITIALIZE_WEIGHTS = _env_bool("THOR_EMBEDDING_PERF_INITIALIZE_WEIGHTS", RANDOMIZE_WEIGHTS)
WEIGHT_INIT_CHUNK_BYTES = int(os.getenv("THOR_EMBEDDING_PERF_WEIGHT_INIT_CHUNK_BYTES", str(64 * 1024 * 1024)))

# NVIDIA profiler hooks.  Enable NVTX ranges for Nsight Systems / Nsight Compute
# range filtering, and enable the CUDA profiler API when using capture ranges.
# Examples:
#   THOR_EMBEDDING_PERF_NVTX=1 nsys profile --trace=cuda,nvtx -o embedding_lookup \
#       pytest -m performance bindings/python/test/core/physical/test_expression_embedding_lookup_performance.py -s
#   THOR_EMBEDDING_PERF_CUDA_PROFILER=1 ncu --profile-from-start off --set full \
#       pytest -m performance bindings/python/test/core/physical/test_expression_embedding_lookup_performance.py -s
#
# Per-launch NVTX ranges are intentionally disabled by default because range
# push/pop overhead can perturb short embedding launches.
#
# For Nsight Compute, prefer hard-DRAM mode so each replayed launch touches more
# data than L2 can comfortably retain:
#   THOR_EMBEDDING_PERF_HARD_DRAM=1 THOR_EMBEDDING_PERF_CUDA_PROFILER=1 \
#   THOR_EMBEDDING_PERF_CASE_FILTER=d768 THOR_EMBEDDING_PERF_DTYPES=fp16 \
#   THOR_EMBEDDING_PERF_INDEX_DTYPES=uint32 THOR_EMBEDDING_PERF_MEASURE_ITERS=4 \
#   ncu --profile-from-start off --target-processes all --section SpeedOfLight \
#       pytest -m performance bindings/python/test/core/physical/test_expression_embedding_lookup_performance.py -s


def _parse_dtype_env(name: str, default: str, allowed: dict[str, thor.DataType]) -> list[thor.DataType]:
    raw = os.getenv(name, default)
    values: list[thor.DataType] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"Unsupported {name} entry {item!r}; allowed values: {sorted(allowed)}")
        values.append(allowed[key])
    return values or list(allowed.values())


DTYPES = _parse_dtype_env(
    "THOR_EMBEDDING_PERF_DTYPES",
    "fp16,bf16,fp32",
    {
        "fp16": thor.DataType.fp16,
        "bf16": thor.DataType.bf16,
        "fp32": thor.DataType.fp32,
    },
)

INDEX_DTYPES = _parse_dtype_env(
    "THOR_EMBEDDING_PERF_INDEX_DTYPES",
    "uint32,uint64",
    {
        "uint32": thor.DataType.uint32,
        "uint64": thor.DataType.uint64,
    },
)


@dataclass(frozen=True)
class EmbeddingLookupPerfCase:
    name: str
    num_tokens: int
    embedding_dim: int
    padding_ratio: float = 0.0
    description: str = ""
    requires_large_opt_in: bool = False


CASES = [
    EmbeddingLookupPerfCase(
        name="d1_tiny_packed_warp",
        num_tokens=1_048_576,
        embedding_dim=1,
        description="Ultra-tiny row width: the generated kernel packs thirty-two independent token lookups into each warp.",
    ),
    EmbeddingLookupPerfCase(
        name="d2_tiny_packed_warp",
        num_tokens=1_048_576,
        embedding_dim=2,
        description="Ultra-tiny row width: the generated kernel packs sixteen independent token lookups into each warp.",
    ),
    EmbeddingLookupPerfCase(
        name="d4_tiny_packed_warp",
        num_tokens=524_288,
        embedding_dim=4,
        description="Ultra-tiny row width: the generated kernel packs eight independent token lookups into each warp.",
    ),
    EmbeddingLookupPerfCase(
        name="d8_tiny_packed_warp",
        num_tokens=262_144,
        embedding_dim=8,
        description="Tiny row width: the generated kernel packs four independent token lookups into each warp.",
    ),
    EmbeddingLookupPerfCase(
        name="d16_tiny_packed_warp",
        num_tokens=262_144,
        embedding_dim=16,
        description="Tiny row width: the generated kernel packs two independent token lookups into each warp.",
    ),
    EmbeddingLookupPerfCase(
        name="d32_many_tokens",
        num_tokens=131_072,
        embedding_dim=32,
        description="Small row width: one logical token group uses a full warp but only one lane loads the row index.",
    ),
    EmbeddingLookupPerfCase(
        name="d128_llm_token_width",
        num_tokens=65_536,
        embedding_dim=128,
        description="Common transformer-ish token embedding width; one warp copies one row in a single iteration.",
    ),
    EmbeddingLookupPerfCase(
        name="d768_hidden_width",
        num_tokens=16_384,
        embedding_dim=768,
        description="BERT/GPT-style hidden width; exercises larger per-lane vector copies without needing multiple warp iterations.",
    ),
    EmbeddingLookupPerfCase(
        name="d128_padding_10pct",
        num_tokens=65_536,
        embedding_dim=128,
        padding_ratio=0.10,
        description="Padding-zero variant; padding rows write zeros and intentionally do not read the table row.",
    ),
    EmbeddingLookupPerfCase(
        name="d2048_single_warp_iteration_limit_fp32",
        num_tokens=8_192,
        embedding_dim=2048,
        description="Hits the one-iteration limit for fp32: 32 lanes x 64 fp32 values per lane.",
        requires_large_opt_in=True,
    ),
    EmbeddingLookupPerfCase(
        name="d4096_multi_iteration",
        num_tokens=4_096,
        embedding_dim=4096,
        description="Forces the warp loader to iterate over a row wider than the one-pass fp32 limit.",
        requires_large_opt_in=True,
    ),
]

if CASE_FILTER:
    CASES = [case for case in CASES if CASE_FILTER in case.name]
elif not ENABLE_LARGE_CASES:
    CASES = [case for case in CASES if not case.requires_large_opt_in]


def _dtype_name(dtype: thor.DataType) -> str:
    return str(dtype).split(".")[-1]


def _bytes_per_element(dtype: thor.DataType) -> int:
    if dtype == thor.DataType.fp32:
        return 4
    if dtype in (thor.DataType.fp16, thor.DataType.bf16):
        return 2
    if dtype == thor.DataType.uint32:
        return 4
    if dtype == thor.DataType.uint64:
        return 8
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _numpy_index_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.uint32:
        return np.uint32
    if dtype == thor.DataType.uint64:
        return np.uint64
    raise AssertionError(f"Unhandled index dtype: {dtype}")


def _tensor_bytes(shape: tuple[int, ...], dtype: thor.DataType) -> int:
    return prod(shape) * _bytes_per_element(dtype)


def _physical_tensor(device_type: DeviceType, shape: tuple[int, ...], dtype: thor.DataType) -> PhysicalTensor:
    device_num = GPU_NUM if device_type == DeviceType.gpu else 0
    return PhysicalTensor(
        Placement(device_type, device_num),
        PhysicalTensor.Descriptor(dtype, list(shape)),
    )


def _gpu_tensor(shape: tuple[int, ...], dtype: thor.DataType) -> PhysicalTensor:
    return _physical_tensor(DeviceType.gpu, shape, dtype)


def _cpu_tensor(shape: tuple[int, ...], dtype: thor.DataType) -> PhysicalTensor:
    return _physical_tensor(DeviceType.cpu, shape, dtype)


def _copy_numpy_to_gpu(values: np.ndarray, stream: Stream, dtype: thor.DataType) -> PhysicalTensor:
    cpu = _cpu_tensor(tuple(int(d) for d in values.shape), dtype)
    view = cpu.numpy()
    assert isinstance(view, np.ndarray)
    view[...] = values
    gpu = _gpu_tensor(tuple(int(d) for d in values.shape), dtype)
    gpu.copy_from_async(cpu, stream)
    return gpu


def _zero_gpu_tensor(shape: tuple[int, ...], dtype: thor.DataType, stream: Stream) -> PhysicalTensor:
    cpu = _cpu_tensor(shape, dtype)
    view = cpu.numpy()
    assert isinstance(view, np.ndarray)
    view.fill(0)
    gpu = _gpu_tensor(shape, dtype)
    gpu.copy_from_async(cpu, stream)
    return gpu


def _random_gpu_tensor(shape: tuple[int, ...], dtype: thor.DataType, stream: Stream, seed: int) -> PhysicalTensor:
    cpu = _cpu_tensor(shape, dtype)
    view = cpu.numpy()
    assert isinstance(view, np.ndarray)
    if len(shape) != 2:
        raise ValueError(f"Expected a rank-2 weight tensor shape, got {shape}")

    rows, cols = shape
    row_bytes = max(1, cols * _bytes_per_element(dtype))
    chunk_rows = max(1, min(rows, WEIGHT_INIT_CHUNK_BYTES // row_bytes))
    rng = np.random.default_rng(seed)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        # Keep values finite and non-zero-ish while avoiding all-zero/all-one pages
        # that can trigger misleading L2 compression results in Nsight Compute.
        values = rng.uniform(-1.0, 1.0, size=(end - start, cols)).astype(np.float32)
        view[start:end, :] = values.astype(view.dtype, copy=False)

    gpu = _gpu_tensor(shape, dtype)
    gpu.copy_from_async(cpu, stream)
    return gpu


def _make_weight_tensor(shape: tuple[int, ...], dtype: thor.DataType, stream: Stream, seed: int) -> PhysicalTensor:
    if RANDOMIZE_WEIGHTS:
        return _random_gpu_tensor(shape, dtype, stream, seed)
    if INITIALIZE_WEIGHTS:
        return _zero_gpu_tensor(shape, dtype, stream)
    return _gpu_tensor(shape, dtype)


def _rotation_stride(pool_slots: int) -> int:
    if pool_slots <= 1:
        return 1
    for candidate in (5, 7, 3, 11, 13, 17, 19):
        if math.gcd(candidate, pool_slots) == 1:
            return candidate
    return 1


def _streamed_bytes_per_launch(
    case: EmbeddingLookupPerfCase,
    value_dtype: thor.DataType,
    index_dtype: thor.DataType,
) -> int:
    value_bytes = _bytes_per_element(value_dtype)
    index_bytes = _bytes_per_element(index_dtype)
    non_padding_tokens = round(case.num_tokens * (1.0 - case.padding_ratio))
    return (
        case.num_tokens * index_bytes
        + non_padding_tokens * case.embedding_dim * value_bytes
        + case.num_tokens * case.embedding_dim * value_bytes
    )


def _choose_pool_slots(bytes_per_launch: int) -> int:
    if EXPLICIT_POOL_SLOTS is not None:
        return max(1, int(EXPLICIT_POOL_SLOTS))
    if HARD_DRAM_PROFILE:
        return max(1, HARD_DRAM_POOL_SLOTS)
    slots = math.ceil(MIN_ROTATING_POOL_BYTES / float(max(1, bytes_per_launch)))
    return max(2, min(MAX_POOL_SLOTS, int(slots)))


def _effective_case(case: EmbeddingLookupPerfCase, value_dtype: thor.DataType) -> EmbeddingLookupPerfCase:
    if EXPLICIT_NUM_TOKENS is not None:
        return replace(case, num_tokens=max(1, int(EXPLICIT_NUM_TOKENS)))

    if not HARD_DRAM_PROFILE:
        return case

    value_bytes = _bytes_per_element(value_dtype)
    row_bytes = max(1, case.embedding_dim * value_bytes)
    non_padding_fraction = max(1.0e-6, 1.0 - case.padding_ratio)
    target_tokens = math.ceil(HARD_DRAM_MIN_TABLE_READ_BYTES / float(row_bytes * non_padding_fraction))

    # Make token counts a multiple of 256 so the one-warp-per-token kernel gets
    # whole 256-thread blocks.  This removes a small source of launch-shape noise
    # from profiler comparisons.
    target_tokens = int(math.ceil(target_tokens / 256.0) * 256)
    if target_tokens <= case.num_tokens:
        return case
    return replace(case, num_tokens=target_tokens)


def _vocabulary_size_for_case(case: EmbeddingLookupPerfCase, pool_slots: int) -> int:
    # Keep each slot's non-padding ids in a distinct row window.  Row zero is
    # reserved as the padding id whenever padding is enabled.
    return 1 + pool_slots * case.num_tokens


def _make_indices_for_slot(
    *,
    case: EmbeddingLookupPerfCase,
    slot: int,
    index_dtype: thor.DataType,
) -> np.ndarray:
    np_dtype = _numpy_index_dtype(index_dtype)
    start = 1 + slot * case.num_tokens
    values = np.arange(start, start + case.num_tokens, dtype=np_dtype)

    # Use distinct rows but shuffled token order.  Sequential ids make the table
    # read look like a simple streaming memcpy, which is too friendly for a real
    # embedding gather.  The slot-specific seed keeps runs deterministic while
    # selecting a different row window per stamped launch.
    rng = np.random.default_rng(0xC0FFEE + 1009 * slot + case.embedding_dim)
    rng.shuffle(values)

    if case.padding_ratio > 0.0:
        stride = max(1, round(1.0 / case.padding_ratio))
        values[::stride] = np_dtype(0)
    return values


def _compile_embedding_lookup(case: EmbeddingLookupPerfCase):
    expr = ex.embedding_lookup(
        ex.input("indices"),
        ex.input("weights"),
        padding_index=0 if case.padding_ratio > 0.0 else None,
    )
    return ex.compile(expr, device_num=GPU_NUM)


def _make_stamped_launch_pool(
    case: EmbeddingLookupPerfCase,
    value_dtype: thor.DataType,
    index_dtype: thor.DataType,
    stream: Stream,
):
    bytes_per_launch = _streamed_bytes_per_launch(case, value_dtype, index_dtype)
    pool_slots = _choose_pool_slots(bytes_per_launch)
    vocabulary_size = _vocabulary_size_for_case(case, pool_slots)

    equation = _compile_embedding_lookup(case)
    weight_shape = (vocabulary_size, case.embedding_dim)
    index_shape = (case.num_tokens,)
    output_shape = (case.num_tokens, case.embedding_dim)

    shared_weights: PhysicalTensor | None = None
    if not ROTATE_WEIGHTS:
        shared_weights = _make_weight_tensor(weight_shape, value_dtype, stream, seed=0xEBD000 + case.embedding_dim)

    launches: list[Callable[[], None]] = []
    slot_inputs: list[dict[str, PhysicalTensor]] = []
    slot_outputs: list[PhysicalTensor] = []

    for slot in range(pool_slots):
        indices_np = _make_indices_for_slot(case=case, slot=slot, index_dtype=index_dtype)
        indices = _copy_numpy_to_gpu(indices_np, stream, index_dtype)
        weights = shared_weights
        if weights is None:
            weights = _make_weight_tensor(
                weight_shape,
                value_dtype,
                stream,
                seed=0xEBD000 + case.embedding_dim * 131 + slot,
            )
        output = _gpu_tensor(output_shape, value_dtype)
        inputs = {"indices": indices, "weights": weights}
        stamped = equation.stamp(inputs, stream, preallocated_output=output)
        runtime_stage_kinds = stamped._debug_stage_kinds()
        assert runtime_stage_kinds == ["EmbeddingLookup"]

        def launch(stamped=stamped, inputs=inputs, output=output) -> None:
            # Capture tensors so their device allocations remain alive for the
            # whole benchmark.  Each launch slot has a distinct index tensor and
            # output tensor; index values select a distinct table row window.
            _ = inputs, output
            stamped.run()

        launches.append(launch)
        slot_inputs.append(inputs)
        slot_outputs.append(output)

    stream.synchronize()
    return {
        "equation": equation,
        "launches": launches,
        "slot_inputs": slot_inputs,
        "slot_outputs": slot_outputs,
        "pool_slots": pool_slots,
        "vocabulary_size": vocabulary_size,
        "weight_shape": weight_shape,
        "output_shape": output_shape,
        "bytes_per_launch": bytes_per_launch,
        "weight_table_bytes_per_slot": _tensor_bytes(weight_shape, value_dtype),
        "output_bytes_per_slot": _tensor_bytes(output_shape, value_dtype),
        "index_bytes_per_slot": _tensor_bytes(index_shape, index_dtype),
    }


def _benchmark_rotating_launches(launches: list[Callable[[], None]], stream: Stream, profile_label: str) -> float:
    assert launches
    pool_slots = len(launches)
    stride = _rotation_stride(pool_slots)

    # Trigger compilation/cache setup for every stamped launch outside timing.
    with PROFILE_HOOKS.range(f"{profile_label}:prime_all_slots"):
        for launch in launches:
            launch()
    stream.synchronize()

    # One more whole-pool sweep catches first cached launch setup while also
    # evicting the row window touched by the first measured launch.
    with PROFILE_HOOKS.range(f"{profile_label}:evict_first_slot_window"):
        for launch in launches:
            launch()
    stream.synchronize()

    warmup_launches = max(WARMUP_ITERS, pool_slots)
    slot = 0
    with PROFILE_HOOKS.range(f"{profile_label}:warmup"):
        for _ in range(warmup_launches):
            launches[slot]()
            slot = (slot + stride) % pool_slots
    stream.synchronize()

    with PROFILE_HOOKS.range(f"{profile_label}:measure"), PROFILE_HOOKS.cuda_profiled_region():
        start = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
        for iteration in range(MEASURE_ITERS):
            if PROFILE_HOOKS.nvtx_launch_ranges:
                with PROFILE_HOOKS.range(f"{profile_label}:measure_launch:{iteration}:slot:{slot}"):
                    launches[slot]()
            else:
                launches[slot]()
            slot = (slot + stride) % pool_slots
        end = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
        elapsed_ms: float = end.synchronize_and_report_elapsed_time_ms(start)
    return elapsed_ms / 1000.0


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("value_dtype", DTYPES, ids=_dtype_name)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=_dtype_name)
def test_embedding_lookup_forward_throughput(
    case: EmbeddingLookupPerfCase,
    value_dtype: thor.DataType,
    index_dtype: thor.DataType,
    record_property,
):
    original_case = case
    case = _effective_case(case, value_dtype)
    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))
    profile_label = f"embedding_lookup:{case.name}:{_dtype_name(value_dtype)}:{_dtype_name(index_dtype)}"
    with PROFILE_HOOKS.range(f"{profile_label}:build_launch_pool"):
        pool = _make_stamped_launch_pool(case, value_dtype, index_dtype, stream)
    launches = pool["launches"]

    # Inspect the compiled stage on the first slot too, so this benchmark also
    # guards that the hot path remains a first-class EmbeddingLookup stage rather
    # than silently becoming a fused scalar fallback.
    compiled_stage_kinds = pool["equation"]._debug_stage_kinds(pool["slot_inputs"][0])
    assert compiled_stage_kinds == ["EmbeddingLookup"]

    elapsed_s = _benchmark_rotating_launches(launches, stream, profile_label)
    ms_per_launch = (elapsed_s / MEASURE_ITERS) * 1_000.0
    launches_per_s = MEASURE_ITERS / elapsed_s
    streamed_gib_per_s = (pool["bytes_per_launch"] * MEASURE_ITERS) / elapsed_s / float(1024**3)
    tokens_per_s = (case.num_tokens * MEASURE_ITERS) / elapsed_s
    output_elements_per_s = (case.num_tokens * case.embedding_dim * MEASURE_ITERS) / elapsed_s
    non_padding_tokens = round(case.num_tokens * (1.0 - case.padding_ratio))
    row_bytes = case.embedding_dim * _bytes_per_element(value_dtype)
    rotating_output_gib = (pool["output_bytes_per_slot"] * pool["pool_slots"]) / float(1024**3)
    rotating_index_gib = (pool["index_bytes_per_slot"] * pool["pool_slots"]) / float(1024**3)
    distinct_weight_row_gib = (non_padding_tokens * row_bytes * pool["pool_slots"]) / float(1024**3)
    weight_table_gib = pool["weight_table_bytes_per_slot"] / float(1024**3)

    record_property("case", case.name)
    record_property("description", case.description)
    record_property("base_num_tokens", original_case.num_tokens)
    record_property("hard_dram_profile", HARD_DRAM_PROFILE)
    record_property("hard_dram_min_table_read_bytes", HARD_DRAM_MIN_TABLE_READ_BYTES)
    record_property("value_dtype", _dtype_name(value_dtype))
    record_property("index_dtype", _dtype_name(index_dtype))
    record_property("num_tokens", case.num_tokens)
    record_property("embedding_dim", case.embedding_dim)
    record_property("row_bytes", row_bytes)
    record_property("padding_ratio", case.padding_ratio)
    record_property("non_padding_tokens", non_padding_tokens)
    record_property("measure_iters", MEASURE_ITERS)
    record_property("warmup_iters", max(WARMUP_ITERS, pool["pool_slots"]))
    record_property("rotating_pool_slots", pool["pool_slots"])
    record_property("rotate_weights", ROTATE_WEIGHTS)
    record_property("initialize_weights", INITIALIZE_WEIGHTS)
    record_property("randomize_weights", RANDOMIZE_WEIGHTS)
    record_property("vocabulary_size", pool["vocabulary_size"])
    record_property("weight_table_gib", weight_table_gib)
    record_property("rotating_output_gib", rotating_output_gib)
    record_property("rotating_index_gib", rotating_index_gib)
    record_property("distinct_weight_row_gib", distinct_weight_row_gib)
    record_property("estimated_streamed_gib_per_launch", pool["bytes_per_launch"] / float(1024**3))
    record_property("ms_per_launch", ms_per_launch)
    record_property("launches_per_second", launches_per_s)
    record_property("estimated_streamed_gib_per_second", streamed_gib_per_s)
    record_property("tokens_per_second", tokens_per_s)
    record_property("output_elements_per_second", output_elements_per_s)
    record_property("compiled_stage_kinds", str(compiled_stage_kinds))
    record_property("runtime_stage_kinds", "['EmbeddingLookup']")
    record_property("nvtx_enabled", PROFILE_HOOKS.nvtx.active)
    record_property("cuda_profiler_enabled", PROFILE_HOOKS.cuda_profiler.active)
    record_property("nvtx_launch_ranges", PROFILE_HOOKS.nvtx_launch_ranges)

    print(
        f"{case.name} [{_dtype_name(value_dtype)}, {_dtype_name(index_dtype)}]: "
        f"{ms_per_launch:.3f} ms/launch | "
        f"{launches_per_s:,.2f} launches/s | "
        f"{tokens_per_s / 1.0e6:,.2f} Mtoken/s | "
        f"{streamed_gib_per_s:,.2f} estimated streamed GiB/s | "
        f"pool={pool['pool_slots']} slots | "
        f"distinct rows in rotation={distinct_weight_row_gib:.2f} GiB | "
        f"table={weight_table_gib:.2f} GiB | "
        f"hard_dram={HARD_DRAM_PROFILE} | "
        f"random_weights={RANDOMIZE_WEIGHTS} | "
        f"rotate_weights={ROTATE_WEIGHTS}"
    )
