#!/usr/bin/env python3
"""SASS size of the dense reducer kernel families, by dtype, from the production objects. Read-only.

Every kernel in the reducer's translation units is reachable from production, so this tool measures the production
objects themselves (fp8_selected_inventory_size.py measures the selected FP8 RK first stages in an isolated archive):

  RKDeepRSmallK.cu.o                        DeepRSmallK and its ARG twins
  RKOneWindow.cu.o                          OneWindow (FP8, FP32, FP16/BF16) and its ARG twins
  KR.cu.o                                   R / KR (FP8, FP32, FP16/BF16; VALUE and ARG)
  RKAlignedK.cu.o                           AlignedK (FP8, FP32, FP16/BF16), the FP8 AlignedK ARG twin, the FP8 and
                                            FP32 ARG continuations
  RKRuntimeK.cu.o                           RuntimeK rotated (FP8, FP32, FP16/BF16) and its ARG twins, RuntimeK
                                            PacketComplete and its ARG twins
  RaggedKRFp8.cu.o                          the ragged KR VALUE kernels for FP8 input
  RaggedKRF16.cu.o                          the same kernels (RaggedKRKernels.cuh) for FP16 / BF16 input
  RaggedKRFp32.cu.o                         the same kernels for FP32 input
  RaggedRKFp8.cu.o                          the ragged RK VALUE kernels for FP8 input
  RaggedRKFp8Narrow.cu.o                    the same kernels at 4 rows per worker, for narrow rows
  RaggedRKF16.cu.o                          the RK kernels for FP16 / BF16 input
  RaggedRKF16Narrow.cu.o                    their narrow-rows kernels
  RaggedRKFp32.cu.o                         the RK kernels for FP32 input
  RaggedRKFp32Narrow.cu.o                   their narrow-rows kernels
  RaggedFinish.cu.o                         the ragged finishing kernels
  RaggedKRArgFp8.cu.o                       the ragged KR ARG kernels for FP8 input (D 1 to 15)
  RaggedKRArgF16.cu.o                       the ragged KR ARG kernels for FP16 / BF16 input
  RaggedKRArgFp32.cu.o                      the ragged KR ARG kernels for FP32 input
  RaggedFinishArg.cu.o                      the ragged ARG finishing kernel
  RaggedFinishArgFp32.cu.o                  the ragged ARG finishing kernel for FP32's pair records
  RaggedRKArgFp8.cu.o                       the ragged RK ARG kernels for FP8 input
  RaggedRKArgF16.cu.o                       the ragged RK ARG kernels for FP16 / BF16 input
  RaggedFinishArgWide.cu.o                  the wide ragged ARG finishing kernels (D above 15)
  RaggedRKArgFp32.cu.o                      the ragged RK ARG kernels for FP32 input
  RaggedFinishArgPairWide.cu.o              the wide ragged ARG finishing kernel for pair records (D above 15)
  StridedR.cu.o                             the StridedR kernels and the stride-0 closed form, the Thor owners of
                                            rank-1 strided views (Reduction/Kernels/StridedReduction.h)

The ragged objects build from Reduction/Ragged/, the others from Reduction/Kernels/. Objects of older builds are
classified but not scanned, so a compare against an older build names them:

  CubReductionSum / Min / Max / Product / L1Norm / SumSquares.cu.o
                                            the value reductions of non-dense views: CUB DeviceReduce /
                                            DeviceSegmentedReduce instantiations and the Thor tiled view kernels, per
                                            op and input dtype
  CubViewLayout.cu.o                        the layout passes of rotated and pitched views (ViewLayout.h)

An instantiation the table does not know is an error.

For every device function in their cubins (readelf FUNC symbols, sizes in bytes of SASS text), it records the kernel,
its input dtype (the first template argument of the shared templates), and:

  owner    FP32 retained | FP8 retained | FP16/BF16 retained |
           FP8+FP32 shared (input-independent, e.g. the KR Split finish) |
           Ragged retained (the merge-path ragged kernels) |
           View strided (the StridedR kernels of rank-1 strided views) |
           View tiled / View CUB / View layout (an older build's tiled view kernels, CUB device reductions of views
           and view layout passes) | device helper
           (the CSV's owner values; they stay as they are, so a compare against an older run groups alike)
  family   RK OneWindow / DeepRSmallK / AlignedK / RuntimeK PacketComplete / RuntimeK Rotated / ARG continuation,
           KR Pack / Vector / Group / Block / Split / Split finish,
           Ragged KR / Ragged KR components / Ragged RK / Ragged finish (ragged retained),
           Tiled direct / pitched / payload transpose / shared transpose (view tiled),
           CUB reduce / CUB reduce tile / CUB segmented (view CUB),
           StridedR kernel / StridedR closed form (view strided),
           Layout row copy / Layout transpose (view layout; the word width, 1-byte / 2-byte / 4-byte, as the dtype)
  kind     VALUE | ARG | carried ARG

Every __global__ kernel declared in the source files must be classified, or the tool fails (classify() is the
one table to extend). Device helpers (noinline functions) are counted per object, not attributed to a kernel.

With --compare=OTHER/device-functions.csv (an earlier run) it also reports, per object and arch, both runs' bytes and
every function whose size changed or that only one run has (written to <out>/compare.txt): a check that a source
move left an object's SASS unchanged. The earlier run is read under today's object and identifier names
(RENAMED_OBJECTS, RENAMED_IDENTIFIERS, SPLIT_OBJECTS), so a moved, renamed or split-off kernel compares with its old
self.

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
import functools
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Sources by path from the source root.
KERNELS = 'Utilities/TensorOperations/Reduction/Kernels/'
RAGGED = 'Utilities/TensorOperations/Reduction/Ragged/'
SOURCES = {
    'RKDeepRSmallK.cu.o': KERNELS + 'RKDeepRSmallK.cu',
    'KR.cu.o': KERNELS + 'KR.cu',
    'RKOneWindow.cu.o': KERNELS + 'RKOneWindow.cu',
    'RKAlignedK.cu.o': KERNELS + 'RKAlignedK.cu',
    'RKRuntimeK.cu.o': KERNELS + 'RKRuntimeK.cu',
    'RaggedKRFp8.cu.o': RAGGED + 'RaggedKRFp8.cu',
    'RaggedKRF16.cu.o': RAGGED + 'RaggedKRF16.cu',
    'RaggedKRFp32.cu.o': RAGGED + 'RaggedKRFp32.cu',
    'RaggedRKFp8.cu.o': RAGGED + 'RaggedRKFp8.cu',
    'RaggedRKFp8Narrow.cu.o': RAGGED + 'RaggedRKFp8Narrow.cu',
    'RaggedRKF16.cu.o': RAGGED + 'RaggedRKF16.cu',
    'RaggedRKF16Narrow.cu.o': RAGGED + 'RaggedRKF16Narrow.cu',
    'RaggedRKFp32.cu.o': RAGGED + 'RaggedRKFp32.cu',
    'RaggedRKFp32Narrow.cu.o': RAGGED + 'RaggedRKFp32Narrow.cu',
    'RaggedFinish.cu.o': RAGGED + 'RaggedFinish.cu',
    'RaggedKRArgFp8.cu.o': RAGGED + 'RaggedKRArgFp8.cu',
    'RaggedKRArgF16.cu.o': RAGGED + 'RaggedKRArgF16.cu',
    'RaggedKRArgFp32.cu.o': RAGGED + 'RaggedKRArgFp32.cu',
    'RaggedFinishArg.cu.o': RAGGED + 'RaggedFinishArg.cu',
    'RaggedFinishArgFp32.cu.o': RAGGED + 'RaggedFinishArgFp32.cu',
    'RaggedRKArgFp8.cu.o': RAGGED + 'RaggedRKArgFp8.cu',
    'RaggedRKArgF16.cu.o': RAGGED + 'RaggedRKArgF16.cu',
    'RaggedFinishArgWide.cu.o': RAGGED + 'RaggedFinishArgWide.cu',
    'RaggedRKArgFp32.cu.o': RAGGED + 'RaggedRKArgFp32.cu',
    'RaggedFinishArgPairWide.cu.o': RAGGED + 'RaggedFinishArgPairWide.cu',
    # Not scanned: the six per-op view-reduction objects of older builds (VIEW_OBJECTS; stale copies may stay in the
    # build tree). Their kernels keep their classification (VIEW_TILED).
    'StridedR.cu.o': KERNELS + 'StridedR.cu',
    # Not scanned: 'CubViewLayout.cu.o', an older build's view layout passes (a stale copy may stay in the build
    # tree). Its kernels keep their classification, so a compare against an older run names them.
}
# An older build's objects whose CUB device-reduction instantiations count as owner View CUB (classified by object;
# not scanned).
VIEW_OBJECTS = ('CubReductionSum.cu.o', 'CubReductionMin.cu.o', 'CubReductionMax.cu.o', 'CubReductionProduct.cu.o',
                'CubReductionL1Norm.cu.o', 'CubReductionSumSquares.cu.o')
# Headers whose __global__ kernels the objects above instantiate (scanned for declarations with the sources).
KERNEL_HEADERS = tuple(RAGGED + h for h in ('RaggedKRKernels.cuh', 'RaggedRKKernels.cuh', 'RaggedKRArgKernels.cuh',
                                             'RaggedRKArgKernels.cuh')) + (KERNELS + 'ReductionOperation.cuh',)
CUDA_GLOBAL_DECLARATION = re.compile(
    r'__global__\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s+)?void\s+(\w+)\s*\(')

DTYPES = {'float': 'FP32', '__nv_fp8_e4m3': 'FP8', '__nv_fp8_e5m2': 'FP8', '__half': 'FP16', '__nv_bfloat16': 'BF16'}

# Kernels templated on the input type: the dtype comes from the first template argument.
SHARED_VALUE = {
    'oneWindowValueKernel': 'RK OneWindow',
    'deepRSmallKValueKernel': 'RK DeepRSmallK',
    'deepRSmallKValueTinyKKernel': 'RK DeepRSmallK',
    'packetCompleteValueKernel': 'RK RuntimeK PacketComplete',
    'krValueGroupKernel': 'KR Group',
    'krValuePackKernel': 'KR Pack',
    'krValueVectorKernel': 'KR Vector',
    'krValueSplitKernel': 'KR Split',
    'krValueBlockKernel': 'KR Block',
    'alignedKValueKernel': 'RK AlignedK',  # FP32 and FP16 / BF16 input
}
FP8_ALIGNED_VALUE = ('fp8AlignedKValueKernel', 'fp8AlignedKValueStagedUnifiedKernel',
                     'fp8AlignedKValueCompleteUnifiedKernel')
# Kernels of one dtype only: (owner, family, kind).
FIXED = {
    'krValueFinishKernel': ('FP8+FP32 shared', 'KR Split finish', 'VALUE'),
    'fp8OneWindowArgKernel': ('FP8 retained', 'RK OneWindow', 'ARG'),
    'fp8DeepRSmallKArgKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8DeepRSmallKArgTinyKKernel': ('FP8 retained', 'RK DeepRSmallK', 'ARG'),
    'fp8RuntimeKRotatedArgKernel': ('FP8 retained', 'RK RuntimeK Rotated', 'ARG'),
    'fp8PacketCompleteArgKernel': ('FP8 retained', 'RK RuntimeK PacketComplete', 'ARG'),
    'fp8AlignedKArgKernel': ('FP8 retained', 'RK AlignedK', 'ARG'),
    'fp8ArgPackedContinuationKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8ArgPackedContinuationSplitKernel': ('FP8 retained', 'RK ARG continuation', 'ARG'),
    'fp8KRArgVectorKernel': ('FP8 retained', 'KR Vector', 'ARG'),
    'alignedKCarriedArgKernel': ('FP32 retained', 'RK AlignedK', 'carried ARG'),
    'runtimeKRotatedCarriedArgKernel': ('FP32 retained', 'RK RuntimeK Rotated', 'carried ARG'),
    'oneWindowCarriedArgKernel': ('FP32 retained', 'RK OneWindow', 'carried ARG'),
    'deepRSmallKCarriedArgKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'deepRSmallKCarriedArgTinyKKernel': ('FP32 retained', 'RK DeepRSmallK', 'carried ARG'),
    'fp32ArgPackedContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgPackedContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'ARG'),
    'fp32ArgWideContinuationKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32ArgWideContinuationSplitKernel': ('FP32 retained', 'RK ARG continuation', 'carried ARG'),
    'fp32KRArgGroupKernel': ('FP32 retained', 'KR Group', 'ARG'),
    'fp32KRArgBlockKernel': ('FP32 retained', 'KR Block', 'ARG'),
    'fp32KRArgPackKernel': ('FP32 retained', 'KR Pack', 'ARG'),
    'fp32KRArgSplitKernel': ('FP32 retained', 'KR Split', 'ARG'),
    'fp32KRArgFinishKernel': ('FP32 retained', 'KR Split finish', 'ARG'),
    'krCarriedArgGroupKernel': ('FP32 retained', 'KR Group', 'carried ARG'),
    'krCarriedArgBlockKernel': ('FP32 retained', 'KR Block', 'carried ARG'),
    'krCarriedArgPackKernel': ('FP32 retained', 'KR Pack', 'carried ARG'),
    'krCarriedArgSplitKernel': ('FP32 retained', 'KR Split', 'carried ARG'),
    'krCarriedArgFinishKernel': ('FP32 retained', 'KR Split finish', 'carried ARG'),
}
# ARG kernels templated on the input type (the dtype comes from the first template argument), as SHARED_VALUE.
SHARED_DTYPE_ARG = {
    'alignedKArgKernel': 'RK AlignedK',
    'runtimeKRotatedArgKernel': 'RK RuntimeK Rotated',
    'packetCompleteArgKernel': 'RK RuntimeK PacketComplete',
    'oneWindowArgKernel': 'RK OneWindow',
    'deepRSmallKArgKernel': 'RK DeepRSmallK',
    'deepRSmallKArgTinyKKernel': 'RK DeepRSmallK',
}
# The R / KR ARG kernels shared by FP8 and FP16 / BF16 through an element-format trait (the first template
# argument: Fp8KRArgFormat or F16KRArgFormat).
SHARED_ARG = {
    'krArgBlockKernel': 'KR Block',
    'krArgGroupKernel': 'KR Group',
    'krArgPackKernel': 'KR Pack',
    'krArgSplitKernel': 'KR Split',
    'krArgFinishKernel': 'KR Split finish',
}
ARG_FORMATS = {'Fp8KRArgFormat': ('FP8 retained', 'FP8'), 'F16KRArgFormat': ('FP16/BF16 retained', 'FP16/BF16')}
# The ragged kernels: (family, kind, dtype); '' takes the dtype from the first template argument (one template per
# kernel for every input dtype).
RAGGED_RETAINED = {
    'raggedKRValueKernel': ('Ragged KR', 'VALUE', ''),  # <InputT, FixedOp>, D = 1
    'raggedKRComponentsKernel': ('Ragged KR components', 'VALUE', ''),  # <InputT, FixedOp>, D >= 2
    'raggedFinishValueKernel': ('Ragged finish', 'VALUE', 'any'),  # <FixedOp>: FP32 boundary records
    'raggedRKValueKernel': ('Ragged RK', 'VALUE', ''),  # <InputT, FixedOp, Rows>, rows >= 16 bytes
    'raggedFinishValueWideKernel': ('Ragged finish', 'VALUE', 'any'),  # <FixedOp>, up to 16 warps per component
    'raggedKRArgKernel': ('Ragged KR', 'ARG', 'key'),  # <Key>, the dtype from the key format
    'raggedKRArgComponentsKernel': ('Ragged KR components', 'ARG', 'key'),  # <Key>, D >= 2
    'raggedFinishArgKernel': ('Ragged finish', 'ARG', 'any'),  # 64-bit candidate records
    'raggedFinishArgPairKernel': ('Ragged finish', 'ARG', 'any'),  # pair records (FP32)
    'raggedRKArgKernel': ('Ragged RK', 'ARG', 'key'),  # <Key, Rows>, rows >= 16 bytes
    'raggedFinishArgWideKernel': ('Ragged finish', 'ARG', 'any'),  # <Transposed>, D above 15
    'raggedFinishArgPairWideKernel': ('Ragged finish', 'ARG', 'any'),  # pair records, D above 15
}
# The ragged ARG kernels' key formats (their first template argument) and their input dtypes.
RAGGED_ARG_KEYS = {'RaggedArgKeyFp8': 'FP8', 'RaggedArgKeyF16': 'FP16/BF16', 'RaggedArgKeyFp32': 'FP32'}
RAGGED_INPUT_DTYPES = ('FP8', 'FP16', 'BF16', 'FP32')
# An older build's Thor tiled view kernels, templated on the input type first: their family. The classification
# stays so a compare against an older run names them.
VIEW_TILED = {
    'directTiledFixedSegmentReductionKernel': 'Tiled direct',
    'pitchedTiledFixedSegmentReductionKernel': 'Tiled pitched',
    'payloadTranspose32x33ReductionKernel': 'Tiled payload transpose',
    'wideComponentTiledSharedTransposeReductionKernel': 'Tiled shared transpose',
}
VIEW_INPUT_DTYPES = ('FP8', 'FP16', 'BF16', 'FP32')
# The StridedR kernels (StridedR.cu): <InputT, Combine>, and the closed form (any input, a runtime dtype).
VIEW_STRIDED = {'stridedRKernel': 'StridedR kernel', 'stridedRClosedFormKernel': 'StridedR closed form'}
# An older build's view layout passes (CubViewLayout.cu), templated on the copied word: its width is the dtype.
VIEW_LAYOUT = {'viewRowCopyKernel': 'Layout row copy', 'viewTransposeKernel': 'Layout transpose'}
LAYOUT_WORDS = {'unsigned char': '1-byte', 'unsigned short': '2-byte', 'unsigned int': '4-byte'}
ROTATED = 'runtimeKRotatedValueKernel'  # <InputT, Op, Transform, Rows, PacketBytes, Warps, Exact>
KNOWN = (set(SHARED_VALUE) | set(SHARED_DTYPE_ARG) | set(SHARED_ARG) | set(FP8_ALIGNED_VALUE) | set(FIXED) |
         set(RAGGED_RETAINED) | set(VIEW_TILED) | set(VIEW_STRIDED) | set(VIEW_LAYOUT) | {ROTATED})


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
    if kernel in VIEW_TILED:
        if dtype not in VIEW_INPUT_DTYPES:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, 'View tiled', VIEW_TILED[kernel], 'VALUE', dtype
    if kernel in VIEW_STRIDED:
        if kernel == 'stridedRClosedFormKernel':
            return kernel, 'View strided', VIEW_STRIDED[kernel], 'VALUE', 'any'
        if dtype not in VIEW_INPUT_DTYPES:
            raise ValueError(f'{kernel} instantiated for an unexpected input type {dtype!r}: {name}')
        return kernel, 'View strided', VIEW_STRIDED[kernel], 'VALUE', dtype
    if kernel in VIEW_LAYOUT:
        word = LAYOUT_WORDS.get(args[0] if args else '')
        if word is None:
            raise ValueError(f'{kernel} instantiated for an unexpected word {args[:1]!r}: {name}')
        return kernel, 'View layout', VIEW_LAYOUT[kernel], 'VALUE', word
    if kernel in RAGGED_RETAINED:
        family, kind, fixed_dtype = RAGGED_RETAINED[kernel]
        if fixed_dtype == 'key':
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
    # Rotated: FP8, the FP32 pass (4 cooperative warps, PrefixReserved) and the 8-warp p16 FP16 / BF16 RuntimeK pass;
    # any other instantiation is an error.
    if len(args) < 6:
        raise ValueError(f'{kernel} with unexpected template arguments: {name}')
    if dtype == 'FP8':
        return kernel, 'FP8 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype == 'FP32' and args[5] == '4':
        return kernel, 'FP32 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    if dtype in ('FP16', 'BF16') and args[4] == '16' and args[5] == '8':
        return kernel, 'FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', dtype
    raise ValueError(f'{kernel} instantiated outside the retained rotated passes: {name}')


def classify_view_cub(name: str, obj_name: str) -> tuple[str, str, str, str, str] | None:
    """A CUB device-reduction kernel instantiated in an older build's view-reduction object, else None."""
    flat = name.removeprefix('void ').lstrip()
    if obj_name not in VIEW_OBJECTS or not flat.startswith('cub::') or not re.search(r'Kernel\w*\s*<', flat):
        return None
    if re.search(r'^cub::\S*EmptyKernel', flat):
        return None
    dtype = next((label for token, label in (('__nv_fp8_e4m3', 'FP8'), ('__nv_fp8_e5m2', 'FP8'), ('__half', 'FP16'),
                                             ('__nv_bfloat16', 'BF16'), ('float const*', 'FP32'),
                                             ('<float>', 'FP32'), ('<float,', 'FP32')) if token in flat), '')
    match = re.match(r'([\w:]*?)(\w+Kernel\w*)\s*<', flat)
    kernel = match.group(2) if match else 'cub kernel'
    family = 'CUB segmented' if 'Segmented' in kernel else 'CUB reduce tile' if 'SingleTile' in kernel else 'CUB reduce'
    return kernel, 'View CUB', family, 'VALUE', dtype


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
    declared = {}
    for obj_name, src in SOURCES.items():
        declared[obj_name] = set(CUDA_GLOBAL_DECLARATION.findall((source / src).read_text()))
    for header in KERNEL_HEADERS:
        declared[header] = set(CUDA_GLOBAL_DECLARATION.findall((source / header).read_text()))
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
                        c = classify(name) or classify_view_cub(name, obj_name)
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


OWNERS = ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained', 'FP8+FP32 shared', 'Ragged retained', 'View tiled',
          'View CUB', 'View strided', 'View layout', 'device helper')


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
        for owner in ('FP32 retained', 'FP8 retained', 'FP16/BF16 retained', 'Ragged retained', 'View tiled',
                      'View CUB', 'View strided', 'View layout'):
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
    lines.append('noinline functions, not attributed to a kernel.')
    lines.append('Ragged retained: the merge-path ragged kernels, in their own objects.')
    lines.append('View strided: the StridedR kernels of rank-1 strided views, StridedR.cu.o; the Split kernel')
    lines.append('finishes in its own launch.')
    lines.append('View tiled / View CUB / View layout: an older build\'s view kernels (the six per-op objects and')
    lines.append('CubViewLayout.cu.o); not scanned, so empty here; a compare against such a build names them.')
    return lines


# The objects and identifiers the kernels' moves renamed. compare() reads the other run through these maps, so a
# moved or renamed kernel compares with its old self.
RENAMED_OBJECTS = {'CubStridedR.cu.o': 'StridedR.cu.o',
                   'CubReductionAwkwardRotatedSharded.cu.o': 'RKRuntimeK.cu.o',
                   'CubReductionKParallelShardedTiled.cu.o': 'RKAlignedK.cu.o',
                   'CubReductionNarrowFlatRCooperative.cu.o': 'RKDeepRSmallK.cu.o',
                   **{f'Cub{n}.cu.o': f'{n}.cu.o' for n in (  # Reduction/Ragged/, without the Cub prefix
                       'RaggedKRFp8', 'RaggedKRF16', 'RaggedKRFp32', 'RaggedRKFp8', 'RaggedRKFp8Narrow',
                       'RaggedRKF16', 'RaggedRKF16Narrow', 'RaggedRKFp32', 'RaggedRKFp32Narrow', 'RaggedFinish',
                       'RaggedKRArgFp8', 'RaggedKRArgF16', 'RaggedKRArgFp32', 'RaggedFinishArg',
                       'RaggedFinishArgFp32', 'RaggedRKArgFp8', 'RaggedRKArgF16', 'RaggedFinishArgWide',
                       'RaggedRKArgFp32', 'RaggedFinishArgPairWide')}}
# Old identifier (kernel, launcher, namespace) or qualified name -> new; whole identifiers only: the FP8 ARG outputs
# (a kernel parameter type, moved out of the anonymous namespace into Reduction/Kernels/KernelCommon.cuh), the kernels'
# family names, the AlignedK kernels' topology argument, the DeepRSmallK kernels' FP8 decoder argument, and the
# namespaces and stage-semantics types without the deleted-backend prefix (kernel arguments and qualifiers).
RENAMED_IDENTIFIERS = {
    'ThorImplementation::CubReductionInternal::(anonymous namespace)::Fp8ArgOutputs':
        'ThorImplementation::ReductionInternal::Fp8ArgOutputs',
    'fp8KRValueGroupKernel': 'krValueGroupKernel',
    'fp8KRValuePackKernel': 'krValuePackKernel',
    'fp8KRValueVectorKernel': 'krValueVectorKernel',
    'fp8KRValueSplitKernel': 'krValueSplitKernel',
    'fp8KRValueFinishKernel': 'krValueFinishKernel',
    'fp8KRValueBlockKernel': 'krValueBlockKernel',
    'fp8KRArgBlockKernel': 'krArgBlockKernel',
    'fp8KRArgGroupKernel': 'krArgGroupKernel',
    'fp8KRArgPackKernel': 'krArgPackKernel',
    'fp8KRArgSplitKernel': 'krArgSplitKernel',
    'fp8KRArgFinishKernel': 'krArgFinishKernel',
    'fp32KRCarriedArgGroupKernel': 'krCarriedArgGroupKernel',
    'fp32KRCarriedArgBlockKernel': 'krCarriedArgBlockKernel',
    'fp32KRCarriedArgPackKernel': 'krCarriedArgPackKernel',
    'fp32KRCarriedArgSplitKernel': 'krCarriedArgSplitKernel',
    'fp32KRCarriedArgFinishKernel': 'krCarriedArgFinishKernel',
    'fp32AlignedKKernel': 'alignedKValueKernel',
    'kParallelFp8PacketKernel': 'fp8AlignedKValueKernel',
    'kParallelFp8StagedHoistedCombineKernel': 'fp8AlignedKValueStagedUnifiedKernel',
    'kParallelFp8CompleteUnifiedKernel': 'fp8AlignedKValueCompleteUnifiedKernel',
    'fp32AlignedKArgKernel': 'alignedKArgKernel',
    'fp32AlignedKCarriedArgKernel': 'alignedKCarriedArgKernel',
    'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'deepRSmallKValueKernel',
    'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel': 'deepRSmallKValueTinyKKernel',
    'fp32DeepRArgKernel': 'deepRSmallKArgKernel',
    'fp32DeepRArgTinyKKernel': 'deepRSmallKArgTinyKKernel',
    'fp8DeepRArgFirstStageKernel': 'fp8DeepRSmallKArgKernel',
    'fp8DeepRArgTinyKFirstStageKernel': 'fp8DeepRSmallKArgTinyKKernel',
    'fp32DeepRCarriedArgKernel': 'deepRSmallKCarriedArgKernel',
    'fp32DeepRCarriedArgTinyKKernel': 'deepRSmallKCarriedArgTinyKKernel',
    'fp8OneWindowRuntimeKCompleteKernel': 'oneWindowValueKernel',
    'fp32OneWindowArgKernel': 'oneWindowArgKernel',
    'fp8OneWindowArgCompleteKernel': 'fp8OneWindowArgKernel',
    'fp32OneWindowCarriedArgKernel': 'oneWindowCarriedArgKernel',
    'awkwardAlignmentRotatedShardedFirstStageKernel': 'runtimeKRotatedValueKernel',
    'fp32RotatedArgKernel': 'runtimeKRotatedArgKernel',
    'fp8RotatedArgKernel': 'fp8RuntimeKRotatedArgKernel',
    'fp32RotatedCarriedArgKernel': 'runtimeKRotatedCarriedArgKernel',
    'fp8GeneralRParallelKTileP16CompleteKernel': 'packetCompleteValueKernel',
    'fp32PacketCompleteArgKernel': 'packetCompleteArgKernel',
    'TiledRKStageTopology': 'RKStageTopology',
    'NarrowFp8TypedFormatDecoder': 'Fp8TypedFormatDecoder',
    'CubReductionInternal': 'ReductionInternal',
    'CubReductionBenchmarking': 'ReductionBenchmarking',
    'CubReductionStageSemantics': 'ReductionStageSemantics',
    'CubReductionStageCombine': 'ReductionStageCombine',
    'CubReductionStageInputTransform': 'ReductionStageInputTransform',
    'CubReductionStageFinalize': 'ReductionStageFinalize',
    'CubReductionStageRole': 'ReductionStageRole',
    'CubReductionPassRole': 'ReductionPassRole',
    'CubKernelOccupancyInfo': 'KernelOccupancyInfo',
    'Fp8OneWindowRuntimeKGeometry': 'OneWindowGeometry',
}
# Objects split off another one, or given some of its functions -> the object the functions were in (under today's
# name: renamed() runs first). compare() reads a function of the other run in the parent object as this run's
# split-off object's when only that object has it, so each object compares with its own functions (device helpers
# both now compile appear in both).
SPLIT_OBJECTS = {'KR.cu.o': 'RKDeepRSmallK.cu.o',
                 'RKRuntimeK.cu.o': 'RKDeepRSmallK.cu.o',
                 'RKAlignedK.cu.o': 'RKDeepRSmallK.cu.o',
                 'RKOneWindow.cu.o': 'RKDeepRSmallK.cu.o'}


@functools.lru_cache(maxsize=None)
def rename_patterns(items: tuple[tuple[str, str], ...]) -> tuple[re.Pattern, re.Pattern | None, dict[str, str]]:
    """renamed()'s patterns for one map: whole identifiers in demangled names, and each plain identifier with its
    length prefix in mangled ones (`20fp8KRValuePackKernel` -> `17krValuePackKernel`; helper rows carry the mangled
    name of their kernel)."""
    identifiers = dict(items)
    words = re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in sorted(identifiers, key=len, reverse=True))
                       + r')(?!\w)')
    mangled = {f'{len(k)}{k}': f'{len(v)}{v}' for k, v in identifiers.items() if re.fullmatch(r'[A-Za-z_]\w*', k)}
    prefixed = (re.compile('|'.join(re.escape(k) for k in sorted(mangled, key=len, reverse=True))) if mangled
                else None)
    return words, prefixed, mangled


def renamed(row: dict, objects: dict[str, str] | None = None, identifiers: dict[str, str] | None = None) -> dict:
    """A row of an earlier run under today's object and identifier names."""
    objects = RENAMED_OBJECTS if objects is None else objects
    identifiers = RENAMED_IDENTIFIERS if identifiers is None else identifiers
    out = dict(row)
    out['object'] = objects.get(out['object'], out['object'])
    if identifiers:
        words, prefixed, mangled = rename_patterns(tuple(sorted(identifiers.items())))
        for key in ('name', 'kernel'):
            if out.get(key):
                out[key] = words.sub(lambda m: identifiers[m.group(1)], out[key])
                if prefixed:
                    out[key] = prefixed.sub(lambda m: mangled[m.group(0)], out[key])
    return out


# File-local symbols carry their translation unit's name and hash in the raw (not demangled) names nvcc gives
# them, e.g. 44_GLOBAL__N__592047fb_11_StridedR_cu_7786666f or 42_INTERNAL_592047fb_11_StridedR_cu_7786666f. A moved
# file renames all of them; compare() reads both runs with the tag set aside.
TU_TAG = re.compile(r'\d+_(GLOBAL__N_|INTERNAL)_[0-9a-f]{8}_\d+_[A-Za-z0-9_]+?_cu_[0-9a-f]{8}')


def split(rows: list[dict], other: list[dict], splits: dict[str, str] | None = None) -> list[dict]:
    """The other run's rows with functions of a parent object moved to the objects split off it. A function
    (name and size) stays in the parent as often as this run has it there; the rest go to a split-off object that has
    it, as often as that object has it."""
    splits = SPLIT_OBJECTS if splits is None else splits
    room = defaultdict(int)
    for r in rows:
        room[(r['object'], r['arch'], r['name'], int(r['bytes']))] += 1
    for r in other:  # the other run's own functions of an object take that object's room first
        if r['object'] not in splits.values() and room[(r['object'], r['arch'], r['name'], int(r['bytes']))] > 0:
            room[(r['object'], r['arch'], r['name'], int(r['bytes']))] -= 1
    out = []
    for r in other:
        key = (r['arch'], r['name'], int(r['bytes']))
        if r['object'] in splits.values():
            for target in [r['object']] + [c for c, parent in splits.items() if parent == r['object']]:
                if room[(target,) + key] > 0:
                    room[(target,) + key] -= 1
                    r = dict(r, object=target)
                    break
        out.append(r)
    return out


# CUDA's internal helpers ($__internal_23_$__cuda_sm3x_div_rn_noftz_f32_slowpath) are numbered per translation
# unit, so a split renumbers those of both objects.
INTERNAL_NUMBER = re.compile(r'\$__internal_\d+_\$')


def tu_neutral(name: str) -> str:
    """A function name with its translation-unit tag and its internal helpers' numbers replaced by fixed ones."""
    return INTERNAL_NUMBER.sub('$__internal_N_$', TU_TAG.sub(lambda m: f'_{m.group(1)}_TU', name))


def compare_moved(rows: list[dict], other: list[dict]) -> list[str]:
    """The same two runs by kernel wherever its object, by (owner, family, kind, dtype), and the device helpers:
    the check for kernels that move between objects (one kernel the same bytes in any object)."""
    lines = []
    for arch in sorted({r['arch'] for r in rows} | {r['arch'] for r in other}):
        def kernels(table):
            out = defaultdict(list)
            for r in table:
                if r['arch'] == arch and r['owner'] != 'device helper':
                    out[r['name']].append(int(r['bytes']))
            return {n: sorted(b) for n, b in out.items()}
        mine, theirs = kernels(rows), kernels(other)
        same = sum(1 for n in mine if theirs.get(n) == mine[n])
        differ = sorted(n for n in mine if n in theirs and theirs[n] != mine[n])
        only_mine, only_theirs = sorted(set(mine) - set(theirs)), sorted(set(theirs) - set(mine))
        lines += [f'# compare by kernel ({arch}): every kernel by name, in whichever object',
                  f'  this {len(mine)} kernels, other {len(theirs)}: {same} identical, {len(differ)} differ, '
                  f'{len(only_mine)} only in this run, {len(only_theirs)} only in the other run']
        for label, names in (('differ', differ), ('only in this run', only_mine),
                             ('only in the other run', only_theirs)):
            for n in names[:20]:
                sizes = [' / '.join(f'{b:,}' for b in t[n]) if n in t else '-' for t in (mine, theirs)]
                lines.append(f'    {label}: this {sizes[0]} other {sizes[1]}: {n[:150]}')

        def groups(table):
            out = defaultdict(int)
            for r in table:
                if r['arch'] == arch and r['owner'] != 'device helper':
                    out[(r['owner'], r['family'], r['kind'], r['dtype'])] += int(r['bytes'])
            return out
        mine_g, theirs_g = groups(rows), groups(other)
        keys = sorted(set(mine_g) | set(theirs_g))
        changed = [k for k in keys if mine_g.get(k) != theirs_g.get(k)]
        lines.append(f'# compare by family ({arch}): bytes per owner, family, kind and dtype')
        lines.append(f'  {len(keys) - len(changed)} of {len(keys)} groups identical')
        for k in changed:
            lines.append(f'    {" / ".join(k)}: this {mine_g.get(k, 0):,} other {theirs_g.get(k, 0):,}')
        helpers = [sum(int(r['bytes']) for r in t if r['arch'] == arch and r['owner'] == 'device helper')
                   for t in (rows, other)]
        lines.append(f'# device helpers ({arch}, noinline functions, counted per object): this {helpers[0]:,} '
                     f'other {helpers[1]:,} ({helpers[0] - helpers[1]:+,})')
    return lines


def compare(rows: list[dict], other: list[dict]) -> list[str]:
    """Per (object, arch) both runs' bytes, and the functions whose size differs or that one run lacks: the other run
    under today's names (renamed()), both runs' names without their translation-unit tags (tu_neutral()), split-off
    objects (split()); then compare_moved()."""
    rows = [dict(r, name=tu_neutral(r['name'])) for r in rows]
    other = split(rows, [dict(r, name=tu_neutral(r['name'])) for r in (renamed(r) for r in other)])
    def index(table):  # a name's sizes (internal helpers repeat a name once their numbers are set aside)
        out = defaultdict(lambda: defaultdict(list))
        for r in table:
            out[(r['object'], r['arch'])][r['name']].append(int(r['bytes']))
        return {k: {n: sorted(b) for n, b in v.items()} for k, v in out.items()}
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
        total_a, total_b = sum(map(sum, a.values())), sum(map(sum, b.values()))
        lines.append(f'  {key[0]:42s} {key[1]:7s} {total_a:>12,} {total_b:>12,} {total_a - total_b:>+10,}  {note}')
        if a and b:
            changes += [(key, n, a.get(n), b.get(n)) for n in changed]
    if changes:
        lines += ['', f'  functions that differ in objects both runs have ({len(changes)}; first 40):']
        for (obj, arch), name, size_a, size_b in changes[:40]:
            size_a, size_b = (' / '.join(map(str, t)) if t else '-' for t in (size_a, size_b))
            lines.append(f'    {obj} {arch} this {size_a} other {size_b}: {name[:160]}')
    return lines + [''] + compare_moved(rows, other)


def self_test():
    ns = 'ThorImplementation::ReductionInternal::(anonymous namespace)::'
    cases = [
        (f'void {ns}oneWindowValueKernel<float, 1, 16, 4, false, 0>(float const*, void*)',
         ('FP32 retained', 'RK OneWindow', 'VALUE', 'FP32')),
        (f'void {ns}oneWindowValueKernel<__nv_fp8_e4m3, 2, 16, 1, false, 2>(__nv_fp8_e4m3 const*)',
         ('FP8 retained', 'RK OneWindow', 'VALUE', 'FP8')),
        (f'void {ns}deepRSmallKValueKernel<float, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 16, 1, 6>(float const*, float*)', ('FP32 retained', 'RK DeepRSmallK', 'VALUE', 'FP32')),
        (f'void {ns}runtimeKRotatedValueKernel<float, {ns}PropagatingMinimumFp32, '
         f'{ns}IdentityFp32, 128, 16, 4, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP32')),
        (f'void {ns}runtimeKRotatedValueKernel<__half, cuda::std::__4::plus<float>, '
         f'{ns}IdentityFp32, 64, 16, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'VALUE', 'FP16')),
        (f'void {ns}alignedKValueKernel<float, 1, 128, ({ns[:-len("(anonymous namespace)::")]}RKStageTopology)0>'
         f'(float const*, void*)', ('FP32 retained', 'RK AlignedK', 'VALUE', 'FP32')),
        (f'void {ns}packetCompleteValueKernel<__half, 2, 3>(__half const*, void*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'VALUE', 'FP16')),
        (f'void {ns}alignedKValueKernel<__nv_bfloat16, 8, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}RKStageTopology)1>(__nv_bfloat16 const*, void*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'VALUE', 'BF16')),
        (f'void {ns}alignedKArgKernel<float, 2, 64, true>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'ARG', 'FP32')),
        (f'void {ns}alignedKArgKernel<__half, 8, 32, false>(__half const*)',
         ('FP16/BF16 retained', 'RK AlignedK', 'ARG', 'FP16')),
        (f'void {ns}runtimeKRotatedArgKernel<__nv_bfloat16, true>(__nv_bfloat16 const*)',
         ('FP16/BF16 retained', 'RK RuntimeK Rotated', 'ARG', 'BF16')),
        (f'void {ns}packetCompleteArgKernel<__half>(__half const*)',
         ('FP16/BF16 retained', 'RK RuntimeK PacketComplete', 'ARG', 'FP16')),
        (f'void {ns}oneWindowArgKernel<__half, 2, 8, true>(__half const*)',
         ('FP16/BF16 retained', 'RK OneWindow', 'ARG', 'FP16')),
        (f'void {ns}deepRSmallKArgTinyKKernel<float, 4>(float const*)',
         ('FP32 retained', 'RK DeepRSmallK', 'ARG', 'FP32')),
        (f'void {ns}runtimeKRotatedArgKernel<float, false>(float const*)',
         ('FP32 retained', 'RK RuntimeK Rotated', 'ARG', 'FP32')),
        (f'void {ns}alignedKCarriedArgKernel<2, 64, true, unsigned long>(float const*)',
         ('FP32 retained', 'RK AlignedK', 'carried ARG', 'FP32')),
        (f'void {ns}krValueFinishKernel<2>(float const*, void*)',
         ('FP8+FP32 shared', 'KR Split finish', 'VALUE', 'FP8+FP32')),
        (f'void {ns}krValueGroupKernel<float, 0, 1>(float const*)', ('FP32 retained', 'KR Group', 'VALUE', 'FP32')),
        (f'void {ns}fp32KRArgSplitKernel(float const*, unsigned long*)', ('FP32 retained', 'KR Split', 'ARG', 'FP32')),
        (f'void {ns}krArgBlockKernel<{ns}Fp8KRArgFormat>(unsigned int const*)',
         ('FP8 retained', 'KR Block', 'ARG', 'FP8')),
        (f'void {ns}krArgFinishKernel<{ns}F16KRArgFormat>(unsigned long const*)',
         ('FP16/BF16 retained', 'KR Split finish', 'ARG', 'FP16/BF16')),
        (f'void {ns}fp8AlignedKValueCompleteUnifiedKernel<__nv_fp8_e5m2, 8ul, 64, '
         f'({ns[:-len("(anonymous namespace)::")]}Fp8CompleteFinalizePolicy)1>(__nv_fp8_e5m2 const*)',
         ('FP8 retained', 'RK AlignedK', 'VALUE', 'FP8')),
    ]
    kr = 'ThorImplementation::ReductionInternal::RaggedKR::'  # the kernels' namespace
    cases += [
        (f'void {kr}raggedKRValueKernel<__nv_fp8_e5m2, 2>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR', 'VALUE', 'FP8')),
        (f'void {ns}raggedFinishValueKernel<0>({ns}RaggedFinishParams)',
         ('Ragged retained', 'Ragged finish', 'VALUE', 'any')),
        (f'void {kr}raggedKRComponentsKernel<__nv_fp8_e4m3, 0>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'FP8')),
        (f'void {kr}raggedKRValueKernel<__half, 0>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR', 'VALUE', 'FP16')),
        (f'void {kr}raggedKRComponentsKernel<__nv_bfloat16, 2>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'BF16')),
        (f'void {kr}raggedKRComponentsKernel<float, 0>({kr}RaggedKRParams)',
         ('Ragged retained', 'Ragged KR components', 'VALUE', 'FP32')),
        (f'void {kr}raggedRKValueKernel<__nv_fp8_e4m3, 2>({kr}RaggedRKParams)',
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP8')),
        (f'void {kr}raggedRKValueKernel<__nv_fp8_e5m2, 0, 4u>({kr}RaggedRKParams)',  # <InputT, FixedOp, Rows>
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP8')),
        (f'void {kr}raggedRKValueKernel<__nv_bfloat16, 2, 4u>({kr}RaggedRKParams)',
         ('Ragged retained', 'Ragged RK', 'VALUE', 'BF16')),
        (f'void {kr}raggedRKValueKernel<float, 0, 2u>({kr}RaggedRKParams)',
         ('Ragged retained', 'Ragged RK', 'VALUE', 'FP32')),
        (f'void {ns}raggedFinishValueWideKernel<0, true>({ns}RaggedFinishParams)',  # <FixedOp, Transposed>
         ('Ragged retained', 'Ragged finish', 'VALUE', 'any')),
        (f'void {kr}raggedKRArgKernel<{kr}RaggedArgKeyFp8>({kr}RaggedKRArgParams)',
         ('Ragged retained', 'Ragged KR', 'ARG', 'FP8')),
        (f'void {kr}raggedKRArgComponentsKernel<{kr}RaggedArgKeyFp8>({kr}RaggedKRArgParams)',
         ('Ragged retained', 'Ragged KR components', 'ARG', 'FP8')),
        (f'void {kr}raggedKRArgKernel<{kr}RaggedArgKeyF16>({kr}RaggedKRArgParams)',
         ('Ragged retained', 'Ragged KR', 'ARG', 'FP16/BF16')),
        (f'void {ns}raggedFinishArgKernel({ns}RaggedFinishArgParams)',
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedKRArgComponentsKernel<{kr}RaggedArgKeyFp32>({kr}RaggedKRArgParams)',
         ('Ragged retained', 'Ragged KR components', 'ARG', 'FP32')),
        (f'void {ns}raggedFinishArgPairKernel({ns}RaggedFinishArgPairParams)',
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyFp8, 4u>({kr}RaggedRKArgParams)',
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP8')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyF16, 2u>({kr}RaggedRKArgParams)',
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP16/BF16')),
        (f'void {ns}raggedFinishArgWideKernel<true>({ns}RaggedFinishArgWideParams)',
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
        (f'void {kr}raggedRKArgKernel<{kr}RaggedArgKeyFp32, 2u>({kr}RaggedRKArgParams)',
         ('Ragged retained', 'Ragged RK', 'ARG', 'FP32')),
        (f'void {ns}raggedFinishArgPairWideKernel({ns}RaggedFinishArgPairWideParams)',
         ('Ragged retained', 'Ragged finish', 'ARG', 'any')),
    ]
    for name, expected in cases:
        got = classify(name)
        assert got is not None and got[1:] == expected, (name, got, expected)
    # The deleted offset-segmented kernels' names do not classify, so one reappearing in an object would be reported
    # as an unclassified Thor kernel (analyze()'s check).
    for legacy in (f'void {ns}narrowVectorOffsetSegmentedReductionKernel<__nv_fp8_e4m3, unsigned int, '
                   f'({ns}VectorSegmentedReductionFamily)0, 8>(__nv_fp8_e4m3 const*, unsigned int const*, void*)',
                   f'void {ns}vectorOffsetSegmentedArgReductionKernel<float, unsigned long, {ns}ArgMinOp, unsigned int>'
                   f'(float const*, unsigned long const*, void*)'):
        assert classify(legacy) is None, legacy
        assert re.search(r'[Kk]ernel\s*<', legacy) and 'ThorImplementation::' in legacy
    assert not any('OffsetSegmented' in obj for obj in SOURCES)
    assert not set(VIEW_OBJECTS) & set(SOURCES)
    assert 'Ragged legacy' not in OWNERS
    # An older build's view-reduction objects' tiled kernels and CUB instantiations, still classified.
    got = classify(f'void {ns}pitchedTiledFixedSegmentReductionKernel<__half, cuda::std::__4::plus<float>, '
                   f'{ns}IdentityFp32, {ns}AdditiveFinalizeFp32>(__half const*, void*)')
    assert got is not None and got[1:] == ('View tiled', 'Tiled pitched', 'VALUE', 'FP16'), got
    try:
        classify(f'void {ns}directTiledFixedSegmentReductionKernel<double, X, Y, Z, 8>(double const*)')
        raise AssertionError('an unexpected tiled view instantiation must fail')
    except ValueError:
        pass
    cub = ('void cub::CUB_300000_SM_1200_NS::detail::reduce::DeviceReduceSingleTileKernel<cub::Policy, '
           f'thrust::transform_iterator<{ns}ToFp32<__nv_bfloat16>, __nv_bfloat16 const*>, float*>(float*)')
    assert classify(cub) is None
    assert classify_view_cub(cub, 'CubReductionSum.cu.o') == (
        'DeviceReduceSingleTileKernel', 'View CUB', 'CUB reduce tile', 'VALUE', 'BF16')
    assert classify_view_cub(cub, 'CubReductionNarrowFlatRCooperative.cu.o') is None
    seg = ('void cub::CUB_300000_SM_1200_NS::detail::reduce::DeviceSegmentedReduceKernel<cub::Policy, '
           f'thrust::transform_iterator<{ns}ToFp32<float>, float const*>, float*>(float const*)')
    assert classify_view_cub(seg, 'CubReductionMax.cu.o')[2:] == ('CUB segmented', 'VALUE', 'FP32')
    assert classify_view_cub('void cub::CUB_300000_SM_1200_NS::detail::EmptyKernel<void>()',
                             'CubReductionMin.cu.o') is None
    # The StridedR kernels.
    got = classify(f'void {ns}stridedRKernel<__nv_fp8_e5m2, 2>(unsigned char const*, unsigned long, unsigned long)')
    assert got is not None and got[1:] == ('View strided', 'StridedR kernel', 'VALUE', 'FP8'), got
    got = classify(f'void {ns}stridedRKernel<float, 0>(float const*, unsigned long, unsigned long)')
    assert got is not None and got[1:] == ('View strided', 'StridedR kernel', 'VALUE', 'FP32'), got
    got = classify(f'void {ns}viewRowCopyKernel<unsigned short, true>(unsigned short const*, unsigned short*)')
    assert got is not None and got[1:] == ('View layout', 'Layout row copy', 'VALUE', '2-byte'), got
    got = classify(f'void {ns}viewTransposeKernel<unsigned char>(unsigned char const*, unsigned char*)')
    assert got is not None and got[1:] == ('View layout', 'Layout transpose', 'VALUE', '1-byte'), got
    try:
        classify(f'void {ns}viewTransposeKernel<double>(double const*)')
        raise AssertionError('a layout kernel over an unexpected word classified')
    except ValueError:
        pass
    got = classify(f'void {ns}stridedRClosedFormKernel(void const*, ThorImplementation::DataType, unsigned long)')
    assert got is not None and got[1:] == ('View strided', 'StridedR closed form', 'VALUE', 'any'), got
    assert classify(f'{ns}stridedRFinalizeOutOfLine(float, float, bool)') is None
    try:
        classify(f'void {ns}stridedRKernel<double, 0>(double const*)')
        raise AssertionError('an unexpected StridedR instantiation must fail')
    except ValueError:
        pass
    helper = f'{ns}finalizeAdditiveFp32OutOfLine(float, float, bool)'
    assert classify(helper) is None
    nested = f'{ns}alignedKValueKernel<float, 1, 32, (X)0>(float const*)::{{lambda()#1}}::operator()() const'
    assert classify(nested) is None
    got = classify(f'void {ns}krValueGroupKernel<__half, 0, 0>(unsigned int const*)')
    assert got is not None and got[1:] == ('FP16/BF16 retained', 'KR Group', 'VALUE', 'FP16'), got
    try:
        classify(f'void {ns}krValueGroupKernel<double, 0, 0>(unsigned int const*)')
        raise AssertionError('an unexpected instantiation of a retained kernel must fail')
    except ValueError:
        pass
    try:  # no 8-warp FP32 rotated pass
        classify(f'void {ns}runtimeKRotatedValueKernel<float, cuda::std::__4::plus<float>, '
                 f'{ns}IdentityFp32, 64, 16, 8, true>(float const*)')
        raise AssertionError('a legacy rotated instantiation must fail')
    except ValueError:
        pass
    try:  # the ragged KR kernels take FP8, FP16, BF16 and FP32
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
    def row(obj, name, size, owner='FP32 retained', family='KR Pack', kernel='k'):
        return {'object': obj, 'arch': 'sm_120', 'name': name, 'bytes': size, 'owner': owner, 'family': family,
                'kind': 'VALUE', 'dtype': 'FP32', 'kernel': kernel}
    old = [row('A.cu.o', 'f', '10'), row('A.cu.o', 'g', '20'), row('B.cu.o', 'h', '5')]
    new = [row('A.cu.o', 'f', 10), row('A.cu.o', 'g', 24), row('B.cu.o', 'h', 5), row('C.cu.o', 'k', 7)]
    text = '\n'.join(compare(new, old))
    assert re.search(r'A\.cu\.o\s+sm_120\s+34\s+30\s+\+4\s+1 differ', text), text
    assert re.search(r'B\.cu\.o\s+sm_120\s+5\s+5\s+\+0\s+identical', text), text
    assert 'new in this run' in text and 'this 24 other 20: g' in text, text
    assert '2 identical, 1 differ, 1 only in this run, 0 only in the other run' in text, text
    # A kernel that moved to another object and was renamed compares with its old self; the object rename maps
    # the old object onto the new one; device helpers are totalled apart.
    old = [row('Old.cu.o', 'void ns::oldKernel<float>(float const*)', '40', kernel='oldKernel'),
           row('Old.cu.o', 'ns::helper(int)', '8', owner='device helper', family='Old', kernel=''),
           row('CubStridedR.cu.o', 's', '9')]
    new = [row('New.cu.o', 'void ns::newKernel<float>(float const*)', 40, kernel='newKernel'),
           row('New.cu.o', 'ns::helper(int)', 8, owner='device helper', family='New', kernel=''),
           row('Other.cu.o', 'ns::helper(int)', 8, owner='device helper', family='Other', kernel=''),
           row('StridedR.cu.o', 's', 9)]
    assert renamed(old[0], {}, {'oldKernel': 'newKernel'})['name'] == new[0]['name']
    assert renamed(old[0], {}, {'oldKernel': 'newKernel'})['kernel'] == 'newKernel'
    assert renamed({'object': 'x', 'name': 'void ns::oldKernelX()', 'kernel': ''}, {}, {'oldKernel': 'n'})['name'] \
        == 'void ns::oldKernelX()'  # whole identifiers only
    # A mangled name (a helper row) is renamed with its identifiers' length prefixes; a qualified key is not.
    mangled = {'object': 'x', 'kernel': '', 'name': '$_ZN2ns36_GLOBAL__N__eda624f4_5_KR_cu_9c23f56120fp8KRValuePack'
                                                   'KernelI6__halfEEvv$__internal_3_$__cuda_sm20_div_u64'}
    assert renamed(mangled, {}, {'fp8KRValuePackKernel': 'krValuePackKernel', 'ns::T': 'ns::U'})['name'] == (
        '$_ZN2ns36_GLOBAL__N__eda624f4_5_KR_cu_9c23f56117krValuePackKernelI6__halfEEvv'
        '$__internal_3_$__cuda_sm20_div_u64')
    assert renamed(dict(mangled, name='$_ZN2ns5oldKr3oldEv'), {}, {'old': 'new'})['name'] == '$_ZN2ns5oldKr3newEv'
    # The OneWindow kernels take the geometry, so an older run's names carry its old name (the real map).
    geometry = {'object': 'RKOneWindow.cu.o', 'kernel': 'oneWindowValueKernel',
                'name': 'void ns::oneWindowValueKernel<float, 1>(ns::Fp8OneWindowRuntimeKGeometry, float)'}
    assert renamed(geometry)['name'] == 'void ns::oneWindowValueKernel<float, 1>(ns::OneWindowGeometry, float)'
    assert renamed(dict(geometry, name='$_ZN2ns20oneWindowValueKernelEN2ns28Fp8OneWindowRuntimeKGeometryEf'))[
        'name'] == '$_ZN2ns20oneWindowValueKernelEN2ns17OneWindowGeometryEf'
    # Each ragged object of an older run (Cub/CubRagged*.cu) maps onto a scanned object of today's.
    assert renamed({'object': 'CubRaggedFinishArgWide.cu.o', 'name': 's', 'kernel': ''})['object'] == \
        'RaggedFinishArgWide.cu.o'
    assert all(v in SOURCES for k, v in RENAMED_OBJECTS.items() if k.startswith('CubRagged')), RENAMED_OBJECTS
    moved = '\n'.join(compare_moved(new, [renamed(r, None, {'oldKernel': 'newKernel'}) for r in old]))
    assert 'this 2 kernels, other 2: 2 identical, 0 differ' in moved, moved
    assert '1 of 1 groups identical' in moved, moved
    assert 'this 16 other 8 (+8)' in moved, moved
    text = '\n'.join(compare(new, old))
    assert re.search(r'StridedR\.cu\.o\s+sm_120\s+9\s+9\s+\+0\s+identical', text), text
    # A moved file's file-local symbols differ only in their translation-unit tag.
    tag_old = ('$_ZN4Thor47_GLOBAL__N__4ee5d666_14_CubStridedR_cu_7786666f1fEv'
               '$_ZN42_INTERNAL_4ee5d666_14_CubStridedR_cu_7786666f1gEv')
    tag_new = ('$_ZN4Thor44_GLOBAL__N__592047fb_11_StridedR_cu_7786666f1fEv'
               '$_ZN42_INTERNAL_592047fb_11_StridedR_cu_7786666f1gEv')
    neutral = '$_ZN4Thor_GLOBAL__N__TU1fEv$_ZN_INTERNAL_TU1gEv'
    assert tu_neutral(tag_old) == tu_neutral(tag_new) == neutral, tu_neutral(tag_old)
    text = '\n'.join(compare([row('StridedR.cu.o', tag_new, 5, owner='device helper')],
                              [row('CubStridedR.cu.o', tag_old, '5', owner='device helper')]))
    assert re.search(r'StridedR\.cu\.o\s+sm_120\s+5\s+5\s+\+0\s+identical', text), text
    # A split. The other run had k and the helper in the parent; this run has k only in the child and the helper
    # in both, and the moved type's qualified name is renamed.
    anon = 'void ns::(anonymous namespace)::'
    old = [row('P.cu.o', 'p', '3'), row('P.cu.o', f'{anon}k(ns::(anonymous namespace)::T)', '7'),
           row('P.cu.o', 'ns::helper(int)', '8', owner='device helper')]
    new = [row('P.cu.o', 'p', 3), row('C.cu.o', f'{anon}k(ns::T)', 7),
           row('P.cu.o', 'ns::helper(int)', 8, owner='device helper'),
           row('C.cu.o', 'ns::helper(int)', 8, owner='device helper')]
    other = split(new, [renamed(r, {}, {'ns::(anonymous namespace)::T': 'ns::T'}) for r in old], {'C.cu.o': 'P.cu.o'})
    assert [r['object'] for r in other] == ['P.cu.o', 'C.cu.o', 'P.cu.o'], other
    moved = '\n'.join(compare_moved(new, other))
    assert 'this 2 kernels, other 2: 2 identical, 0 differ' in moved, moved
    assert 'this 16 other 8 (+8)' in moved, moved
    # CUDA's internal helpers renumbered by the split; each object keeps its own copies.
    old = [row('P.cu.o', '$__internal_7_$__cuda_div', '20', owner='device helper'),
           row('P.cu.o', '$__internal_9_$__cuda_div', '24', owner='device helper')]
    new = [row('P.cu.o', '$__internal_3_$__cuda_div', 20, owner='device helper'),
           row('C.cu.o', '$__internal_1_$__cuda_div', 24, owner='device helper')]
    assert tu_neutral(old[0]['name']) == tu_neutral(new[1]['name']) == '$__internal_N_$__cuda_div'
    rows_new = [dict(r, name=tu_neutral(r['name'])) for r in new]
    other = split(rows_new, [dict(r, name=tu_neutral(r['name'])) for r in old], {'C.cu.o': 'P.cu.o'})
    assert [r['object'] for r in other] == ['P.cu.o', 'C.cu.o'], other
    # An object that existed before the split keeps its own copies; only the parent's extra copies move to it.
    old = [row('P.cu.o', '$__internal_1_$__cuda_div', '20', owner='device helper'),
           row('P.cu.o', '$__internal_2_$__cuda_div', '20', owner='device helper'),
           row('C.cu.o', '$__internal_1_$__cuda_div', '20', owner='device helper'),
           row('D.cu.o', '$__internal_1_$__cuda_div', '20', owner='device helper')]
    new = [row('P.cu.o', '$__internal_1_$__cuda_div', 20, owner='device helper'),
           row('C.cu.o', '$__internal_1_$__cuda_div', 20, owner='device helper'),
           row('D.cu.o', '$__internal_1_$__cuda_div', 20, owner='device helper'),
           row('D.cu.o', '$__internal_2_$__cuda_div', 20, owner='device helper')]
    rows_new = [dict(r, name=tu_neutral(r['name'])) for r in new]
    other = split(rows_new, [dict(r, name=tu_neutral(r['name'])) for r in old],
                  {'C.cu.o': 'P.cu.o', 'D.cu.o': 'P.cu.o'})
    assert [r['object'] for r in other] == ['P.cu.o', 'D.cu.o', 'C.cu.o', 'D.cu.o'], other
    print(f'self-test: {len(cases) + 24} classifications, the summary and the comparisons OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source', type=Path, help='the Thor source root (.. from the build directory)')
    parser.add_argument('--out', type=Path, default=Path('dense-sass'))
    parser.add_argument('--compare', type=Path, help="an earlier run's device-functions.csv")
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
