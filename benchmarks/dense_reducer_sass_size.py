#!/usr/bin/env python3
"""SASS size of the dense reducer kernel families, by dtype, from the production objects (1232). Read-only.

1118's fp8_selected_inventory_size.py measured the selected FP8 RK first stages in an isolated archive. That
isolation was needed while the production translation units still compiled unselected FP8 experiments. Today every
retained kernel in them is reachable, so this tool measures the production objects themselves:

  CubReductionNarrowFlatRCooperative.cu.o   OneWindow, DeepRSmallK, RuntimeK PacketComplete, ARG twins, R / KR
  CubReductionKParallelShardedTiled.cu.o    AlignedK (FP8, FP32, FP16/BF16), FP32 ARG continuations
  CubReductionAwkwardRotatedSharded.cu.o    RuntimeK rotated (FP8, FP32, FP16/BF16), rotated ARG
  CubRaggedKRFp8.cu.o                       (1265) the retained ragged KR VALUE kernels for FP8 input
  CubRaggedKRF16.cu.o                       (1284) the same kernels (RaggedKRKernels.cuh) for FP16 / BF16 input
  CubRaggedKRFp32.cu.o                      (1284) the same kernels for FP32 input
  CubRaggedRKFp8.cu.o                       (1285) the retained ragged RK VALUE kernels for FP8 input
  CubRaggedRKFp8Narrow.cu.o                 (1288) the same kernels at 4 rows per worker, for narrow rows
  CubRaggedRKF16.cu.o                       (1295) the RK kernels for FP16 / BF16 input
  CubRaggedRKF16Narrow.cu.o                 (1295) their narrow-rows kernels
  CubRaggedRKFp32.cu.o                      (1295) the RK kernels for FP32 input
  CubRaggedRKFp32Narrow.cu.o                (1295) their narrow-rows kernels
  CubRaggedFinish.cu.o                      (1265) the retained ragged finishing kernels
  CubRaggedKRArgFp8.cu.o                    (1298) the retained ragged KR ARG kernels for FP8 input (1300: D 1 to 15)
  CubRaggedKRArgF16.cu.o                    (1311) the retained ragged KR ARG kernels for FP16 / BF16 input
  CubRaggedKRArgFp32.cu.o                   (1313) the retained ragged KR ARG kernels for FP32 input
  CubRaggedFinishArg.cu.o                   (1298) the retained ragged ARG finishing kernel
  CubRaggedFinishArgFp32.cu.o               (1313) the ragged ARG finishing kernel for FP32's pair records
  CubRaggedRKArgFp8.cu.o                    (1314) the retained ragged RK ARG kernels for FP8 input
  CubRaggedRKArgF16.cu.o                    (1320) the retained ragged RK ARG kernels for FP16 / BF16 input
  CubRaggedFinishArgWide.cu.o               (1314) the wide ragged ARG finishing kernels (D above 15)
  CubRaggedRKArgFp32.cu.o                   (1321) the retained ragged RK ARG kernels for FP32 input
  CubRaggedFinishArgPairWide.cu.o           (1321) the wide ragged ARG finishing kernel for pair records (D above 15)

1259 deleted the legacy KParallel / RCooperative kernels (and CubReductionCooperativeShardedTiled.cu), so every
kernel in these objects is retained; an instantiation the table does not know is an error. 1330 (ragged phase 9)
deleted the legacy offset-segmented objects (CubReductionOffsetSegmented.cu.o, CubArgReductionOffsetSegmented.cu.o:
1263-1329's `Ragged legacy` owner, 450 kernels and 4.03 MB at sm_120 with 0.21 MB of device helpers), so that owner
is gone.

For every device function in their cubins (readelf FUNC symbols, sizes in bytes of SASS text), it records the kernel,
its input dtype (the first template argument of the shared templates), and:

  owner    FP32 retained | FP8 retained | FP16/BF16 retained (1233) |
           FP8+FP32 shared (input-independent, e.g. the KR Split finish) |
           Ragged retained (1265: the merge-path ragged kernels; 1330: the only ragged kernels) |
           device helper
  family   RK OneWindow / DeepRSmallK / AlignedK / RuntimeK PacketComplete / RuntimeK Rotated / ARG continuation,
           KR Pack / Vector / Group / Block / Split / Split finish,
           Ragged KR / Ragged KR components / Ragged RK / Ragged finish (ragged retained)
  kind     VALUE | ARG | carried ARG

Every __global__ kernel declared in the source files must be classified, or the tool fails (classify() is the
one table to extend). Device helpers (noinline functions) are counted per object, not attributed to a kernel.

With --compare=OTHER/device-functions.csv (1265: an earlier run) it also reports, per object and arch, both runs' bytes
and every function whose size changed or that only one run has (written to <out>/compare.txt): a check that a source
move left an object's SASS unchanged.

Usage (from the build directory):
  python3 ../benchmarks/dense_reducer_sass_size.py --build . --source .. --out dense-sass
  python3 ../benchmarks/dense_reducer_sass_size.py --build . --source .. --out dense-sass --compare OLD/device-functions.csv
  python3 ../benchmarks/dense_reducer_sass_size.py --self-test
Writes <out>/device-functions.csv and <out>/summary.txt and prints the summary. Requires cuobjdump, readelf and
c++filt on PATH (CUDA toolkit and binutils).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SOURCES = {
    'CubReductionNarrowFlatRCooperative.cu.o': 'CubReductionNarrowFlatRCooperative.cu',
    'CubReductionKParallelShardedTiled.cu.o': 'CubReductionKParallelShardedTiled.cu',
    'CubReductionAwkwardRotatedSharded.cu.o': 'CubReductionAwkwardRotatedSharded.cu',
    'CubRaggedKRFp8.cu.o': 'CubRaggedKRFp8.cu',  # 1265
    'CubRaggedKRF16.cu.o': 'CubRaggedKRF16.cu',  # 1284
    'CubRaggedKRFp32.cu.o': 'CubRaggedKRFp32.cu',  # 1284
    'CubRaggedRKFp8.cu.o': 'CubRaggedRKFp8.cu',  # 1285
    'CubRaggedRKFp8Narrow.cu.o': 'CubRaggedRKFp8Narrow.cu',  # 1288
    'CubRaggedRKF16.cu.o': 'CubRaggedRKF16.cu',  # 1295
    'CubRaggedRKF16Narrow.cu.o': 'CubRaggedRKF16Narrow.cu',  # 1295
    'CubRaggedRKFp32.cu.o': 'CubRaggedRKFp32.cu',  # 1295
    'CubRaggedRKFp32Narrow.cu.o': 'CubRaggedRKFp32Narrow.cu',  # 1295
    'CubRaggedFinish.cu.o': 'CubRaggedFinish.cu',  # 1265
    'CubRaggedKRArgFp8.cu.o': 'CubRaggedKRArgFp8.cu',  # 1298
    'CubRaggedKRArgF16.cu.o': 'CubRaggedKRArgF16.cu',  # 1311
    'CubRaggedKRArgFp32.cu.o': 'CubRaggedKRArgFp32.cu',  # 1313
    'CubRaggedFinishArg.cu.o': 'CubRaggedFinishArg.cu',  # 1298
    'CubRaggedFinishArgFp32.cu.o': 'CubRaggedFinishArgFp32.cu',  # 1313
    'CubRaggedRKArgFp8.cu.o': 'CubRaggedRKArgFp8.cu',  # 1314
    'CubRaggedRKArgF16.cu.o': 'CubRaggedRKArgF16.cu',  # 1320
    'CubRaggedFinishArgWide.cu.o': 'CubRaggedFinishArgWide.cu',  # 1314
    'CubRaggedRKArgFp32.cu.o': 'CubRaggedRKArgFp32.cu',  # 1321
    'CubRaggedFinishArgPairWide.cu.o': 'CubRaggedFinishArgPairWide.cu',  # 1321
}
# 1284: headers whose __global__ kernels the objects above instantiate (scanned for declarations with the sources).
KERNEL_HEADERS = ('RaggedKRKernels.cuh', 'RaggedRKKernels.cuh', 'RaggedKRArgKernels.cuh', 'RaggedRKArgKernels.cuh')
CUDA_GLOBAL_DECLARATION = re.compile(
    r'__global__\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s+)?void\s+(\w+)\s*\(')

DTYPES = {'float': 'FP32', '__nv_fp8_e4m3': 'FP8', '__nv_fp8_e5m2': 'FP8', '__half': 'FP16', '__nv_bfloat16': 'BF16'}

# Kernels templated on the input type: the dtype comes from the first template argument.
SHARED_VALUE = {
    'fp8OneWindowRuntimeKCompleteKernel': 'RK OneWindow',
    'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'RK DeepRSmallK',
    'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel': 'RK DeepRSmallK',
    'fp8GeneralRParallelKTileP16CompleteKernel': 'RK RuntimeK PacketComplete',
    'fp8KRValueGroupKernel': 'KR Group',
    'fp8KRValuePackKernel': 'KR Pack',
    'fp8KRValueVectorKernel': 'KR Vector',
    'fp8KRValueSplitKernel': 'KR Split',
    'fp8KRValueBlockKernel': 'KR Block',
    'fp32AlignedKKernel': 'RK AlignedK',  # 1243: FP32 and FP16 / BF16 input
}
FP8_ALIGNED_VALUE = ('kParallelFp8PacketKernel', 'kParallelFp8StagedHoistedCombineKernel',
                     'kParallelFp8CompleteUnifiedKernel')
# Kernels of one dtype only: (owner, family, kind).
FIXED = {
    'fp8KRValueFinishKernel': ('FP8+FP32 shared', 'KR Split finish', 'VALUE'),
    'fp8OneWindowArgCompleteKernel': ('FP8 retained', 'RK OneWindow', 'ARG'),
    'fp8DeepRArgFirstStageKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8DeepRArgTinyKFirstStageKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8RotatedArgKernel': ('FP8 retained', 'RK RuntimeK Rotated', 'ARG'),
    'fp8PacketCompleteArgKernel': ('FP8 retained', 'RK RuntimeK PacketComplete', 'ARG'),
    'fp8AlignedKArgKernel': ('FP8 retained', 'RK AlignedK', 'ARG'),
    'fp8ArgPackedContinuationKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8ArgPackedContinuationSplitKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8KRArgVectorKernel': ('FP8 retained', 'KR Vector', 'ARG'),
    'fp32AlignedKCarriedArgKernel': ('FP32 retained', 'RK AlignedK', 'carried ARG'),
    'fp32RotatedCarriedArgKernel': ('FP32 retained', 'RK RuntimeK Rotated', 'carried ARG'),
    'fp32OneWindowCarriedArgKernel': ('FP32 retained', 'RK OneWindow', 'carried ARG'),
    'fp32DeepRCarriedArgKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'fp32DeepRCarriedArgTinyKKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'fp32ArgPackedContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgPackedContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgWideContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32ArgWideContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32KRArgGroupKernel': ('FP32 retained', 'KR Group', 'ARG'),
    'fp32KRArgBlockKernel': ('FP32 retained', 'KR Block', 'ARG'),
    'fp32KRArgPackKernel': ('FP32 retained', 'KR Pack', 'ARG'),
    'fp32KRArgSplitKernel': ('FP32 retained', 'KR Split', 'ARG'),
    'fp32KRArgFinishKernel': ('FP32 retained', 'KR Split finish', 'ARG'),
    'fp32KRCarriedArgGroupKernel': ('FP32 retained', 'KR Group', 'carried ARG'),
    'fp32KRCarriedArgBlockKernel': ('FP32 retained', 'KR Block', 'carried ARG'),
    'fp32KRCarriedArgPackKernel': ('FP32 retained', 'KR Pack', 'carried ARG'),
    'fp32KRCarriedArgSplitKernel': ('FP32 retained', 'KR Split', 'carried ARG'),
    'fp32KRCarriedArgFinishKernel': ('FP32 retained', 'KR Split finish', 'carried ARG'),
}
# 1248: ARG kernels templated on the input type (the dtype comes from the first template argument), as SHARED_VALUE.
SHARED_DTYPE_ARG = {
    'fp32AlignedKArgKernel': 'RK AlignedK',
    'fp32RotatedArgKernel': 'RK RuntimeK Rotated',  # 1249
    'fp32PacketCompleteArgKernel': 'RK RuntimeK PacketComplete',  # 1249
    'fp32OneWindowArgKernel': 'RK OneWindow',  # 1250
    'fp32DeepRArgKernel': 'RK DeepRSmallK',  # 1250
    'fp32DeepRArgTinyKKernel': 'RK DeepRSmallK',  # 1250
}
# 1237: the R / KR ARG kernels shared by FP8 and FP16 / BF16 through an element-format trait (the first template
# argument: Fp8KRArgFormat or F16KRArgFormat).
SHARED_ARG = {
    'fp8KRArgBlockKernel': 'KR Block',
    'fp8KRArgGroupKernel': 'KR Group',
    'fp8KRArgPackKernel': 'KR Pack',
    'fp8KRArgSplitKernel': 'KR Split',
    'fp8KRArgFinishKernel': 'KR Split finish',
}
ARG_FORMATS = {'Fp8KRArgFormat': ('FP8 retained', 'FP8'), 'F16KRArgFormat': ('FP16/BF16 retained', 'FP16/BF16')}
# 1263-1329: RAGGED_LEGACY (the offset-segmented vector kernels) and RAGGED_OBJECTS (their objects' CUB
# DeviceSegmentedReduce instantiations), deleted with the kernels in 1330.
# 1265: the retained ragged kernels: (family, kind, dtype); '' takes the dtype from the first template argument.
# 1284: one template per kernel for every input dtype (1265-1283: raggedKRFp8ValueKernel, raggedKRFp8ComponentsKernel).
RAGGED_RETAINED = {
    'raggedKRValueKernel': ('Ragged KR', 'VALUE', ''),  # <InputT, FixedOp>, D = 1
    'raggedKRComponentsKernel': ('Ragged KR components', 'VALUE', ''),  # 1280: <InputT, FixedOp>, D >= 2
    'raggedFinishValueKernel': ('Ragged finish', 'VALUE', 'any'),  # <FixedOp>: FP32 boundary records
    'raggedRKValueKernel': ('Ragged RK', 'VALUE', ''),  # 1285: <InputT, FixedOp>, rows >= 16 bytes; 1288: <.., Rows>
    'raggedFinishValueWideKernel': ('Ragged finish', 'VALUE', 'any'),  # 1291: <FixedOp>, up to 16 warps per component
    'raggedKRArgKernel': ('Ragged KR', 'ARG', 'key'),  # 1298: <Key>, the dtype from the key format
    'raggedKRArgComponentsKernel': ('Ragged KR components', 'ARG', 'key'),  # 1300: <Key>, D >= 2
    'raggedFinishArgKernel': ('Ragged finish', 'ARG', 'any'),  # 1298: 64-bit candidate records
    'raggedFinishArgPairKernel': ('Ragged finish', 'ARG', 'any'),  # 1313: pair records (FP32)
    'raggedRKArgKernel': ('Ragged RK', 'ARG', 'key'),  # 1314: <Key, Rows>, rows >= 16 bytes
    'raggedFinishArgWideKernel': ('Ragged finish', 'ARG', 'any'),  # 1314: <Transposed>, D above 15
    'raggedFinishArgPairWideKernel': ('Ragged finish', 'ARG', 'any'),  # 1321: pair records, D above 15
}
# 1298: the ragged ARG kernels' key formats (their first template argument) and their input dtypes.
RAGGED_ARG_KEYS = {'RaggedArgKeyFp8': 'FP8', 'RaggedArgKeyF16': 'FP16/BF16', 'RaggedArgKeyFp32': 'FP32'}  # 1311, 1313
RAGGED_INPUT_DTYPES = ('FP8', 'FP16', 'BF16', 'FP32')  # 1284 (1265-1283: FP8)
ROTATED = 'awkwardAlignmentRotatedShardedFirstStageKernel'  # <InputT, Op, Transform, Rows, PacketBytes, Warps, Exact>
KNOWN = (set(SHARED_VALUE) | set(SHARED_DTYPE_ARG) | set(SHARED_ARG) | set(FP8_ALIGNED_VALUE) | set(FIXED) |
         set(RAGGED_RETAINED) | {ROTATED})


def template_arguments(name: str, kernel: str) -> list[str] | None:
    """Top-level template arguments of `kernel<...>` in a demangled name, or None when it has none."""
    match = re.search(r'(?:::|\b)' + re.escape(kernel) + r'\s*<', name)
    if not match:
        return None
    depth, start, args = 0, match.end(), []
    for i in range(match.end() - 1, len(name)):
        ch = name[i]
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
            if depth == 0:
                args.append(name[start:i].strip())
                return args
        elif ch == ',' and depth == 1:
            args.append(name[start:i].strip())
            start = i + 1
    return None


def kernel_of(name: str) -> str | None:
    """The known kernel this demangled function is, or None (a device helper or an unknown function)."""
    flat = name.replace('(anonymous namespace)', 'anonymous')
    # A function nested in a kernel (an out-of-line lambda, `kernel<...>(...)::{lambda...}`) is a helper.
    if re.search(r'\)\s*(?:const\s*)?::', flat):
        return None
    for kernel in KNOWN:
        if re.search(r'(?:::|\b)' + re.escape(kernel) + r'\s*[<(]', flat):
            return kernel
    return None


def classify(name: str) -> tuple[str, str, str, str, str] | None:
    """(kernel, owner, family, kind, dtype) for a kernel's demangled name; None for a device helper."""
    kernel = kernel_of(name)
    if kernel is None:
        return None
    args = template_arguments(name, kernel) or []
    dtype = DTYPES.get(args[0], args[0]) if args else ''
    if kernel in RAGGED_RETAINED:
        family, kind, fixed_dtype = RAGGED_RETAINED[kernel]
        if fixed_dtype == 'key':  # 1298
            key = args[0].split('::')[-1] if args else ''
            if key not in RAGGED_ARG_KEYS:
                raise ValueError(f'{kernel} instantiated for an unexpected key format {key!r}: {name}')
            return kernel, 'Ragged retained', family, kind, RAGGED_ARG_KEYS[key]
        if not fixed_dtype and dtype not in RAGGED_INPUT_DTYPES:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, 'Ragged retained', family, kind, fixed_dtype or dtype
    if kernel in FIXED:
        owner, family, kind = FIXED[kernel]
        return kernel, owner, family, kind, {'FP32 retained': 'FP32', 'FP8 retained': 'FP8'}.get(owner, 'FP8+FP32')
    if kernel in SHARED_ARG:
        trait = args[0].split('::')[-1] if args else ''
        if trait not in ARG_FORMATS:
            raise ValueError(f'{kernel} instantiated for an unexpected format {trait!r}: {name}')
        owner, dtype = ARG_FORMATS[trait]
        return kernel, owner, SHARED_ARG[kernel], 'ARG', dtype
    if kernel in SHARED_DTYPE_ARG:
        owner = {'FP32': 'FP32 retained', 'FP16': 'FP16/BF16 retained', 'BF16': 'FP16/BF16 retained'}.get(dtype)
        if owner is None:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, owner, SHARED_DTYPE_ARG[kernel], 'ARG', dtype
    if kernel in SHARED_VALUE:
        owner = {'FP32': 'FP32 retained', 'FP8': 'FP8 retained', 'FP16': 'FP16/BF16 retained',
                 'BF16': 'FP16/BF16 retained'}.get(dtype)
        if owner is None:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, owner, SHARED_VALUE[kernel], 'VALUE', dtype
    if kernel in FP8_ALIGNED_VALUE:
        if dtype != 'FP8':
            raise ValueError(f'{kernel} instantiated for a non-FP8 input type {dtype!r}: {name}')
        return kernel, 'FP8 retained', 'RK AlignedK', 'VALUE', dtype
    # Rotated: FP8, the retained FP32 pass (4 cooperative warps, PrefixReserved) and (1245) the 8-warp p16 FP16 / BF16
    # RuntimeK pass. 1259 deleted the legacy 8-warp RCooperative rotated pass, so any other instantiation is an error.
    if len(args) < 6:
        raise ValueError(f'{kernel} with unexpected template arguments: {name}')
    if dtype == 'FP8':
        return kernel, 'FP8 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype == 'FP32' and args[5] == '4':
        return kernel, 'FP32 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype in ('FP16', 'BF16') and args[4] == '16' and args[5] == '8':
        return kernel, 'FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    raise ValueError(f'{kernel} instantiated outside the retained rotated passes: {name}')


def run(cmd, cwd=None, text_input=None):
    result = subprocess.run(cmd, cwd=cwd, input=text_input, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def func_symbols(readelf_output: str) -> list[tuple[str, int]]:
    """(mangled name, size) of every defined FUNC symbol with a size (readelf -W -s)."""
    seen, out = set(), []
    for line in readelf_output.splitlines():
        parts = line.strip().split(None, 7)
        if len(parts) != 8 or not parts[0].endswith(':') or parts[3] != 'FUNC' or parts[6] in ('UND', 'ABS'):
            continue
        try:
            size = int(parts[2], 0)  # readelf prints sizes above 99999 in hex
        except ValueError:
            continue
        name = parts[7].split()[-1].split('@@', 1)[0]
        if size and (name, size) not in seen:
            seen.add((name, size))
            out.append((name, size))
    return out


def find_object(build: Path, name: str) -> Path:
    found = sorted(build.rglob(name))
    preferred = [p for p in found if 'ThorCubCudaObjects' in str(p)]
    if not (preferred or found):
        sys.exit(f'no {name} under {build} (build Thor first)')
    return (preferred or found)[0].resolve()


def analyze(build: Path, source: Path, out: Path, other_csv: Path | None = None):
    for program in ('cuobjdump', 'readelf', 'c++filt'):
        if shutil.which(program) is None:
            sys.exit(f'{program} not found on PATH')
    cub = source / 'Utilities/TensorOperations/Cub'
    declared = {}
    for obj_name, src in SOURCES.items():
        declared[obj_name] = set(CUDA_GLOBAL_DECLARATION.findall((cub / src).read_text()))
    for header in KERNEL_HEADERS:  # 1284
        declared[header] = set(CUDA_GLOBAL_DECLARATION.findall((cub / header).read_text()))
    unknown = sorted(set().union(*declared.values()) - KNOWN)
    if unknown:
        sys.exit(f'__global__ kernels without a classification (extend classify()): {unknown}')
    out.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for obj_name in SOURCES:
        obj = find_object(build, obj_name)
        with tempfile.TemporaryDirectory() as tmp:
            run(['cuobjdump', '-xelf', 'all', str(obj)], cwd=tmp)
            cubins = sorted(Path(tmp).rglob('*.cubin'))
            if not cubins:
                sys.exit(f'no device ELF in {obj}')
            for cubin in cubins:
                arch = re.search(r'sm_?(\d+)', cubin.name)
                arch = f'sm_{arch.group(1)}' if arch else cubin.name
                symbols = func_symbols(run(['readelf', '-W', '-s', str(cubin)]))
                names = run(['c++filt', '-n'], text_input='\n'.join(s for s, _ in symbols) + '\n').splitlines()
                if len(names) != len(symbols):
                    sys.exit(f'demangle mismatch in {cubin}')
                for (mangled, size), name in zip(symbols, names):
                    try:
                        c = classify(name)
                    except ValueError as error:
                        problems.append(str(error))
                        continue
                    if c is None:
                        kernel_like = bool(re.search(r'[Kk]ernel\s*<', name)) and 'ThorImplementation::' in name
                        if kernel_like:
                            problems.append(f'unclassified Thor kernel in {obj_name}: {name}')
                        c = ('', 'device helper', obj_name.removesuffix('.cu.o'), '', '')
                    kernel, owner, family, kind, dtype = c
                    rows.append({'object': obj_name, 'arch': arch, 'owner': owner, 'family': family, 'kind': kind,
                                 'dtype': dtype, 'bytes': size, 'kernel': kernel, 'name': name})
    with (out / 'device-functions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['object'])
        writer.writeheader()
        writer.writerows(rows)
    lines = summarize(rows)
    if other_csv is not None:
        with other_csv.open(newline='') as stream:
            comparison = compare(rows, list(csv.DictReader(stream)))
        (out / 'compare.txt').write_text('\n'.join(comparison) + '\n')
        lines += [''] + comparison
    if problems:
        lines += ['', f'ERROR: {len(problems)} functions could not be classified:'] + [f'  {p}' for p in problems[:40]]
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if problems:
        sys.exit(1)


OWNERS = ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained', 'FP8+FP32 shared', 'Ragged retained', 'device helper')


def summarize(rows: list[dict]) -> list[str]:
    lines = []
    for arch in sorted({r['arch'] for r in rows}):
        here = [r for r in rows if r['arch'] == arch]
        lines.append(f'# {arch}: SASS text bytes (readelf FUNC sizes) of the dense and ragged reducer objects')
        totals = defaultdict(lambda: [0, 0])
        for r in here:
            key = (r['owner'], r['family'], r['kind'])
            totals[key][0] += 1
            totals[key][1] += r['bytes']
        owner_totals = defaultdict(lambda: [0, 0])
        for (owner, _, _), (n, b) in totals.items():
            owner_totals[owner][0] += n
            owner_totals[owner][1] += b
        lines.append(f'  {"owner":17s} {"family":30s} {"kind":12s} {"kernels":>7s} {"bytes":>12s}')
        for owner in OWNERS:
            keys = sorted(k for k in totals if k[0] == owner)
            for key in keys:
                n, b = totals[key]
                count = '' if owner == 'device helper' else str(n)
                lines.append(f'  {key[0]:17s} {key[1]:30s} {key[2]:12s} {count:>7s} {b:>12,}')
            if keys:
                n, b = owner_totals[owner]
                count = '' if owner == 'device helper' else str(n)
                lines.append(f'  {owner + " total":60s} {count:>7s} {b:>12,}')
                lines.append('')
        grand = sum(b for _, b in owner_totals.values())
        lines.append(f'  {"all functions in these objects":60s} {"":>7s} {grand:>12,}')
        for owner in ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained', 'Ragged retained'):
            if owner not in owner_totals:
                continue
            by_family = defaultdict(int)
            for (o, family, _), (_, b) in totals.items():
                if o == owner:
                    by_family[family.split(' ')[0] + ' ' + family.split(' ')[1]] += b
            parts = ', '.join(f'{f} {b / 1e6:.2f} MB' for f, b in sorted(by_family.items()))
            lines.append(f'  {owner}: {owner_totals[owner][1] / 1e6:.2f} MB ({parts})')
        lines.append('')
    lines.append('FP8+FP32 shared: input-independent kernels both dtypes run (counted once). Device helpers are')
    lines.append('noinline functions, not attributed to a kernel. The legacy dense families were deleted in 1259.')
    lines.append('Ragged retained: the merge-path ragged kernels (1265 on), their own objects; the legacy')
    lines.append('offset-segmented kernels (1263-1329, 4.03 MB at sm_120) were deleted in 1330.')
    lines.append('1118\'s isolated FP8 RK first-stage inventory measured 8.80 MB.')
    return lines


def compare(rows: list[dict], other: list[dict]) -> list[str]:
    """1265: per (object, arch) both runs' bytes, and the functions whose size differs or that one run lacks."""
    def index(table):
        out = defaultdict(dict)
        for r in table:
            out[(r['object'], r['arch'])][r['name']] = int(r['bytes'])
        return out
    mine, theirs = index(rows), index(other)
    lines = ['# compare: SASS bytes this run / the other run, per object and arch',
             f'  {"object":42s} {"arch":7s} {"this":>12s} {"other":>12s} {"delta":>10s}  functions']
    changes = []
    for key in sorted(set(mine) | set(theirs)):
        a, b = mine.get(key, {}), theirs.get(key, {})
        changed = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
        note = 'identical' if not changed else f'{len(changed)} differ'
        if not a:
            note = 'only in the other run'
        elif not b:
            note = 'new in this run'
        total_a, total_b = sum(a.values()), sum(b.values())
        lines.append(f'  {key[0]:42s} {key[1]:7s} {total_a:>12,} {total_b:>12,} {total_a - total_b:>+10,}  {note}')
        if a and b:
            changes += [(key, n, a.get(n), b.get(n)) for n in changed]
    if changes:
        lines += ['', f'  functions that differ in objects both runs have ({len(changes)}; first 40):']
        for (obj, arch), name, size_a, size_b in changes[:40]:
            lines.append(f'    {obj} {arch} this {size_a if size_a is not None else "-"} other '
                         f'{size_b if size_b is not None else "-"}: {name[:160]}')
    return lines


def self_test():
    ns = 'ThorImplementation::CubReductionInternal::(anonymous namespace)::'
    cases = [
        (f'void {ns}fp8OneWindowRuntimeKCompleteKernel<float, 1, 16, 4, false, 0>(float const*, void*)',
         ('FP32 retained', 'RK OneWindow', 'VALUE', 'FP32')),
        (f'void {ns}fp8OneWindowRuntimeKCompleteKernel<__nv_fp8_e4m3, 2, 16, 1, false, 2>(__nv_fp8_e4m3 const*)',
         ('FP8 retained', 'RK OneWindow', 'VALUE', 'FP8')),
        (f'void {ns}narrowFp8PackedFlatRCooperativeFirstStageKernel<float, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 16, 1, 6>(float const*, float*)', ('FP32 retained', 'RK DeepRSmallK', 'VALUE', 'FP32')),
        (f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<float, {ns}PropagatingMinimumFp32, '
         f'{ns}IdentityFp32, 128, 16, 4, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP32')),
        (f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<__half, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 64, 16, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP16')),  # 1245
        (f'void {ns}fp32AlignedKKernel<float, 1, 128, ({ns[:-len("(anonymous namespace)::")]}TiledRKStageTopology)0>'
         f'(float const*, void*)', ('FP32 retained', 'RK AlignedK', 'VALUE', 'FP32')),
        (f'void {ns}fp8GeneralRParallelKTileP16CompleteKernel<__half, 2, 3>(__half const*, void*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'VALUE', 'FP16')),  # 1244
        (f'void {ns}fp32AlignedKKernel<__nv_bfloat16, 8, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}TiledRKStageTopology)1>(__nv_bfloat16 const*, void*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'VALUE', 'BF16')),
        (f'void {ns}fp32AlignedKArgKernel<float, 2, 64, true>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'ARG', 'FP32')),
        (f'void {ns}fp32AlignedKArgKernel<__half, 8, 32, false>(__half const*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'ARG', 'FP16')),  # 1248
        (f'void {ns}fp32RotatedArgKernel<__nv_bfloat16, true>(__nv_bfloat16 const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'ARG', 'BF16')),  # 1249
        (f'void {ns}fp32PacketCompleteArgKernel<__half>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'ARG', 'FP16')),  # 1249
        (f'void {ns}fp32OneWindowArgKernel<__half, 2, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK OneWindow', 'ARG', 'FP16')),  # 1250
        (f'void {ns}fp32DeepRArgTinyKKernel<float, 4>(float const*)',
         ('FP32 retained', 'RK DeepRSmallK', 'ARG', 'FP32')),
        (f'void {ns}fp32RotatedArgKernel<float, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'ARG', 'FP32')),
        (f'void {ns}fp32AlignedKCarriedArgKernel<2, 64, true, unsigned long>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'carried ARG', 'FP32')),
        (f'void {ns}fp8KRValueFinishKernel<2>(float const*, void*)',
         ('FP8+FP32 shared', 'KR Split finish', 'VALUE', 'FP8+FP32')),
        (f'void {ns}fp8KRValueGroupKernel<float, 0, 1>(float const*)', ('FP32 retained', 'KR Group', 'VALUE', 'FP32')),
        (f'void {ns}fp32KRArgSplitKernel(float const*, unsigned long*)', ('FP32 retained', 'KR Split', 'ARG', 'FP32')),
        (f'void {ns}fp8KRArgBlockKernel<{ns}Fp8KRArgFormat>(unsigned int const*)',
         ('FP8 retained', 'KR Block', 'ARG', 'FP8')),
        (f'void {ns}fp8KRArgFinishKernel<{ns}F16KRArgFormat>(unsigned long const*)',
         ('FP16/BF16 retained', 'KR Split finish', 'ARG', 'FP16/BF16')),
        (f'void {ns}kParallelFp8CompleteUnifiedKernel<__nv_fp8_e5m2, 8ul, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}Fp8CompleteFinalizePolicy)1>(__nv_fp8_e5m2 const*)',
         ('FP8 retained', 'RK AlignedK', 'VALUE', 'FP8')),
    ]
    kr = 'ThorImplementation::CubReductionInternal::RaggedKR::'  # 1284: the kernels' namespace
    cases += [  # 1265
        (f'void {kr}raggedKRValueKernel<__nv_fp8_e5m2, 2>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR', 'VALUE', 'FP8')),
        (f'void {ns}raggedFinishValueKernel<0>({ns}RaggedFinishParams)',
         ('Ragged retained', 'Ragged finish', 'VALUE', 'any')),
        (f'void {kr}raggedKRComponentsKernel<__nv_fp8_e4m3, 0>({kr}RaggedKRParams)',  # 1280
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'FP8')),
        (f'void {kr}raggedKRValueKernel<__half, 0>({kr}RaggedKRParams)',  # 1284
         ('Ragged retained', 'Ragged KR', 'VALUE', 'FP16')),
        (f'void {kr}raggedKRComponentsKernel<__nv_bfloat16, 2>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'BF16')),
        (f'void {kr}raggedKRComponentsKernel<float, 0>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'FP32')),
        (f'void {kr}raggedRKValueKernel<__nv_fp8_e4m3, 2>({kr}RaggedRKParams)',  # 1285
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP8')),
        (f'void {kr}raggedRKValueKernel<__nv_fp8_e5m2, 0, 4u>({kr}RaggedRKParams)',  # 1288: <InputT, FixedOp, Rows>
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP8')),
        (f'void {kr}raggedRKValueKernel<__nv_bfloat16, 2, 4u>({kr}RaggedRKParams)',  # 1295
         ('Ragged retained', 'Ragged RK', 'VALUE', 'BF16')),
        (f'void {kr}raggedRKValueKernel<float, 0, 2u>({kr}RaggedRKParams)',  # 1295
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP32')),
        (f'void {ns}raggedFinishValueWideKernel<0, true>({ns}RaggedFinishParams)',  # 1294: <FixedOp, Transposed>
         ('Ragged retained', 'Ragged finish', 'VALUE', 'any')),
        (f'void {kr}raggedKRArgKernel<{kr}RaggedArgKeyFp8>({kr}RaggedKRArgParams)',  # 1298
         ('Ragged retained', 'Ragged KR', 'ARG', 'FP8')),
        (f'void {kr}raggedKRArgComponentsKernel<{kr}RaggedArgKeyFp8>({kr}RaggedKRArgParams)',  # 1300
         ('Ragged retained', 'Ragged KR components', 'ARG', 'FP8')),
        (f'void {kr}raggedKRArgKernel<{kr}RaggedArgKeyF16>({kr}RaggedKRArgParams)',  # 1311
         ('Ragged retained', 'Ragged KR', 'ARG', 'FP16/BF16')),
        (f'void {ns}raggedFinishArgKernel({ns}RaggedFinishArgParams)',  # 1298
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedKRArgComponentsKernel<{kr}RaggedArgKeyFp32>({kr}RaggedKRArgParams)',  # 1313
         ('Ragged retained', 'Ragged KR components', 'ARG', 'FP32')),
        (f'void {ns}raggedFinishArgPairKernel({ns}RaggedFinishArgPairParams)',  # 1313
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyFp8, 4u>({kr}RaggedRKArgParams)',  # 1314
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP8')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyF16, 2u>({kr}RaggedRKArgParams)',  # 1320
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP16/BF16')),
        (f'void {ns}raggedFinishArgWideKernel<true>({ns}RaggedFinishArgWideParams)',  # 1314
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyFp32, 2u>({kr}RaggedRKArgParams)',  # 1321
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP32')),
        (f'void {ns}raggedFinishArgPairWideKernel({ns}RaggedFinishArgPairWideParams)',  # 1321
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
    ]
    for name, expected in cases:
        got = classify(name)
        assert got is not None and got[1:] == expected, (name, got, expected)
    # 1330: the legacy offset-segmented kernels are deleted: their names no longer classify, so one reappearing in an
    # object would be reported as an unclassified Thor kernel (analyze()'s check).
    for legacy in (f'void {ns}narrowVectorOffsetSegmentedReductionKernel<__nv_fp8_e4m3, unsigned int, '
                   f'({ns}VectorSegmentedReductionFamily)0, 8>(__nv_fp8_e4m3 const*, unsigned int const*, void*)',
                   f'void {ns}vectorOffsetSegmentedArgReductionKernel<float, unsigned long, {ns}ArgMinOp, unsigned int>'
                   f'(float const*, unsigned long const*, void*)'):
        assert classify(legacy) is None, legacy
        assert re.search(r'[Kk]ernel\s*<', legacy) and 'ThorImplementation::' in legacy
    assert not any('OffsetSegmented' in obj for obj in SOURCES)
    assert 'Ragged legacy' not in OWNERS
    helper = f'{ns}finalizeAdditiveFp32OutOfLine(float, float, bool)'
    assert classify(helper) is None
    nested = f'{ns}fp32AlignedKKernel<float, 1, 32, (X)0>(float const*)::{{lambda()#1}}::operator()() const'
    assert classify(nested) is None
    got = classify(f'void {ns}fp8KRValueGroupKernel<__half, 0, 0>(unsigned int const*)')  # 1233
    assert got is not None and got[1:] == ('FP16/BF16 retained', 'KR Group', 'VALUE', 'FP16'), got
    try:
        classify(f'void {ns}fp8KRValueGroupKernel<double, 0, 0>(unsigned int const*)')
        raise AssertionError('an unexpected instantiation of a retained kernel must fail')
    except ValueError:
        pass
    try:  # 1259: the legacy 8-warp FP32 rotated pass is deleted
        classify(f'void {ns}awkwardAlignmentRotatedShardedFirstStageKernel<float, cuda::std::__4::plus<float>, '
                 f'{ns}IdentityFp32, 64, 16, 8, true>(float const*)')
        raise AssertionError('a legacy rotated instantiation must fail')
    except ValueError:
        pass
    try:  # 1284: the ragged KR kernels take FP8, FP16, BF16 and FP32 (1265-1283: FP8 only)
        classify(f'void {kr}raggedKRValueKernel<double, 0>({kr}RaggedKRParams)')
        raise AssertionError('an unexpected ragged KR instantiation must fail')
    except ValueError:
        pass
    assert classify(f'{ns}raggedMeanOutOfLine(float, unsigned long)') is None
    rows = [{'arch': 'sm_120', 'owner': c[1][0], 'family': c[1][1], 'kind': c[1][2], 'bytes': 100} for c in cases]
    rows.append({'arch': 'sm_120', 'owner': 'device helper', 'family': 'CubReductionNarrowFlatRCooperative',
                 'kind': '', 'bytes': 50})
    summary = summarize(rows)
    assert any('FP32 retained total' in line for line in summary)
    assert any('Ragged retained: ' in line for line in summary)
    old = [{'object': 'A.cu.o', 'arch': 'sm_120', 'name': 'f', 'bytes': '10'},
           {'object': 'A.cu.o', 'arch': 'sm_120', 'name': 'g', 'bytes': '20'},
           {'object': 'B.cu.o', 'arch': 'sm_120', 'name': 'h', 'bytes': '5'}]
    new = [{'object': 'A.cu.o', 'arch': 'sm_120', 'name': 'f', 'bytes': 10},
           {'object': 'A.cu.o', 'arch': 'sm_120', 'name': 'g', 'bytes': 24},
           {'object': 'B.cu.o', 'arch': 'sm_120', 'name': 'h', 'bytes': 5},
           {'object': 'C.cu.o', 'arch': 'sm_120', 'name': 'k', 'bytes': 7}]
    text = '\n'.join(compare(new, old))
    assert re.search(r'A\.cu\.o\s+sm_120\s+34\s+30\s+\+4\s+1 differ', text), text
    assert re.search(r'B\.cu\.o\s+sm_120\s+5\s+5\s+\+0\s+identical', text), text
    assert 'new in this run' in text and 'this 24 other 20: g' in text, text
    print(f'self-test: {len(cases) + 13} classifications, the summary and the comparison OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source', type=Path, help='the Thor source root (.. from the build directory)')
    parser.add_argument('--out', type=Path, default=Path('dense-sass'))
    parser.add_argument('--compare', type=Path, help="an earlier run's device-functions.csv (1265)")
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.build is None or args.source is None:
        parser.error('--build and --source are required (or --self-test)')
    analyze(args.build, args.source, args.out, args.compare)


if __name__ == '__main__':
    main()
