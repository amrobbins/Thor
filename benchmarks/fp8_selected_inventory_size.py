#!/usr/bin/env python3
"""Build-only, isolated *selected* FP8 first-stage kernel census for Thor.

This tool does not modify reduction sources. `generate` copies just the definitions
from three production CUDA translation units into build-directory translation
units and explicitly references every selected kernel instantiation. CMake
compiles them with Thor's CUDA flags into a standalone archive.

`analyze` extracts device ELFs from the isolated objects and counts exact
per-function SASS sizes, separate from object/archive bytes. No GPU run needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

DTYPES = (('__nv_fp8_e4m3', 'E4M3'), ('__nv_fp8_e5m2', 'E5M2'))
# The frozen selector's ten OneWindow variants (Fp8OneWindowPolicy::candidates). Since P5-J2 they run bucketed
# runtime-K units (ONE_WINDOW_UNITS); the P4/P8 variants have none.
ONE_WINDOW = ((1,1,16),(1,2,16),(1,4,16),(1,8,16),(2,1,16),(4,1,16),(8,1,16),
              (8,1,4),(4,1,8),(8,1,8))
# OneWindow reduction operations are compile-time specializations. The final
# output conversion and additive finalization remain runtime; none of these
# kernels multiplies the inventory by input transform or output dtype. Since
# P5-F MAX launches the MIN kernel (FixedOp 2) on sign-folded FP8 bytes.
ONE_WINDOW_OPERATIONS = ((0, 'SUM'), (1, 'PRODUCT'), (2, 'MIN-MAX'))
# RuntimeK retains four compiled FP32 combines for its Packet and two Rotated
# first-stage topologies; input FP8 format is not selected at runtime. Packet
# compiles all three transforms for every combine. P5-F: rotated MIN/MAX fold
# the transform into the FP8 bytes (identity kernels only); PrefixReserved MAX
# also folds its sign and launches the MIN kernel, Exact keeps MIN and MAX.
# Runtime shard/warp/tile geometry adds no CUDA symbols.
RUNTIME_K_OPERATIONS = (
    (0, 'SUM', 'cuda::std::plus<float>'),
    (1, 'PRODUCT', 'cuda::std::multiplies<float>'),
    (2, 'MIN', 'PropagatingMinimumFp32'),
    (3, 'MAX', 'PropagatingMaximumFp32'),
)
# AlignedK keeps separate typed PRODUCT/MIN/MAX transformed kernels.
ALIGNED_TYPED_OPERATIONS = RUNTIME_K_OPERATIONS[1:]
EXPECTED_COUNTS = {'OneWindow': 90, 'DeepRSmallK': 264,
                   'AlignedK': 164, 'RuntimeK': 204}
# OneWindow units (outputs per warp, packet bytes, components-per-lane bucket, load-ahead); mirrors
# ONE_WINDOW_RUNTIME_K_UNITS (P5-J2). x 3 physical operations (MIN/MAX folded) x 2 FP8 formats = 90 kernels.
ONE_WINDOW_UNITS = (
    (1, 16, 1, False), (1, 16, 2, True), (1, 16, 4, False), (1, 16, 8, True), (1, 16, 16, True),
    (2, 16, 1, False), (2, 16, 2, False), (2, 16, 4, False), (2, 16, 8, True),
    (4, 16, 2, False), (4, 16, 4, False), (4, 16, 8, True),
    (8, 16, 1, False), (8, 16, 2, False), (8, 16, 4, False),
)
EXPERIMENTAL_FAMILIES = ()  # families measured but excluded from the selected total (none since P5-J2)
DEEP_ROWS = (64,128,256,512,1024)
ROTATED_ROWS = (32,64,128,256,512,1024)
ALIGNED_STAGED = ((2,32),(2,64),(2,128),(8,32),(8,64),(8,128))
ALIGNED_COMPLETE = ((2,32),(8,32),(8,64))

FAMILIES = {
    'fp8OneWindowRuntimeKCompleteKernel': 'OneWindow',
    'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel': 'DeepRSmallK',
    'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'DeepRSmallK',
    'kParallelFp8PacketKernel': 'AlignedK',
    'kParallelFp8StagedHoistedCombineKernel': 'AlignedK',
    'kParallelFp8CompleteUnifiedKernel': 'AlignedK',
    'fp8GeneralRParallelKTileP16CompleteKernel': 'RuntimeK',
    'awkwardAlignmentRotatedShardedFirstStageKernel': 'RuntimeK',
}


# Names of first-party __global__ kernels are extracted from the same build-local
# source TUs we compile, rather than guessed from a broad "Kernel<" regex.
# This distinguishes an accidentally instantiated Thor kernel from CUDA/CCCL
# support kernels whose names legitimately contain "kernel<".
CUDA_GLOBAL_DECLARATION = re.compile(
    # One level of nested parentheses inside __launch_bounds__(...), e.g. a parenthesized min-blocks expression.
    r'__global__\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s+)?void\s+(\w+)\s*\('
)


def first_party_kernel_names(objects: list[Path], archive: Path) -> set[str]:
    generated_dir = archive.parent
    kernels: set[str] = set()
    for name in ('narrow', 'aligned', 'rotated'):
        path = generated_dir / f'{name}.cu'
        if not path.is_file():
            raise SystemExit(f'Missing generated CUDA source for inventory membership check: {path}')
        kernels.update(CUDA_GLOBAL_DECLARATION.findall(path.read_text()))
    unknown_selected = set(FAMILIES) - kernels
    if unknown_selected:
        raise SystemExit(f'Selected kernel declarations absent from generated sources: {sorted(unknown_selected)}')
    return kernels


def is_named_kernel_function(name: str, kernel_name: str) -> bool:
    # Match an actual demangled function name (not a template parameter type).
    # Thor kernels are in the CubReductionInternal namespace; the name can
    # include an anonymous namespace qualifier as well. A function nested in a
    # kernel (an out-of-line lambda: `kernel<...>(...)::{lambda(...)#1}::operator()`)
    # is a device helper, not the kernel.
    if re.search(r'\)\s*(?:const\s*)?::', name.replace('(anonymous namespace)', 'anonymous')):
        return False
    return bool(re.search(r'(?:::|\b)' + re.escape(kernel_name) + r'\s*[<(]', name))


def specialization_key(kernel: str, args: str) -> str:
    """Normalized `kernel<args>` for matching manifest entries against demangled names."""
    text = re.sub(r'\s+', '', f'{kernel}<{args}>')
    return re.sub(r'(?<=\d)(?:ul|u|l)\b', '', text)


def cut_unique(text: str, marker: str, source: str) -> str:
    count = text.count(marker)
    if count != 1:
        raise ValueError(f'{source}: expected one {marker!r}, got {count}. Refusing source extraction.')
    return text.split(marker, 1)[0]




def validate_policy_snapshot(source: Path):
    """Fail closed if the inventory literals drift from the policy headers."""
    b = source / 'Utilities/TensorOperations/Cub/Fp8RK'
    ow = (b / 'Fp8OneWindowSelector.h').read_text()
    marker = 'inline constexpr std::array<Variant, 10> candidates = '
    if ow.count(marker) != 1:
        raise ValueError('OneWindow inventory declaration changed')
    variant_line = ow.split(marker,1)[1].split('}}};',1)[0]
    entries = tuple(tuple(map(int, a)) for a in re.findall(r'(\d+),(\d+),(\d+)',variant_line))
    if entries != ONE_WINDOW:
        raise ValueError('OneWindow variants changed')
    if 'return k >= 1 && (k <= 497 || k == 512);' not in ow:
        raise ValueError('OneWindow K coverage changed')
    deep = (b/'Fp8DeepRSmallKSelector.h').read_text()
    if f'compiled_rows = {{{", ".join(map(str,DEEP_ROWS))}}}' not in deep:
        raise ValueError('DeepRSmallK rows changed')
    rt = (b/'Fp8RuntimeKSelector.h').read_text()
    if f'rotated_row_depths = {{{", ".join(map(str,ROTATED_ROWS))}}}' not in rt:
        raise ValueError('RuntimeK rows changed')
    aligned = (b/'Fp8AlignedKSelector.h').read_text()
    required = ('packet_bytes = {2, 8}', 'cta_threads = {32, 64, 128}',
                'return (packet == 2 && threads == 32)',
                '(packet == 8 && (threads == 32 || threads == 64))')
    if any(x not in aligned for x in required):
        raise ValueError('AlignedK inventory changed')
    narrow = (source / 'Utilities/TensorOperations/Cub/CubReductionNarrowFlatRCooperative.cu').read_text()
    table = narrow.split('ONE_WINDOW_RUNTIME_K_UNITS{{', 1)[-1].split('}};', 1)[0]
    units = [(int(o), int(p), int(c), la == 'true')
             for o, p, c, la in re.findall(r'\{(\d+), (\d+), (\d+), (true|false)\}', table)]
    if len(units) != len(set(units)) or tuple(sorted(units)) != tuple(sorted(ONE_WINDOW_UNITS)):
        raise ValueError(f'OneWindow units changed: {sorted(units)}')


def read_prune_groups(path: Path | None):
    if path is None:
        return set()
    groups = set()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'dtype', 'operation', 'family', 'physical_kernel'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Prune-spec CSV missing required compiled-group columns')
        for row in reader:
            if row['family'] != 'aligned_k' or not row['operation'].endswith(('_abs', '_square')):
                raise ValueError('Accepted prune list must contain ONLY transformed AlignedK kernels')
            key = (row['dtype'], row['operation'], row['physical_kernel'])
            if key in groups:
                raise ValueError(f'Duplicate accepted prune key: {key}')
            groups.add(key)
    return groups


def generate(source: Path, out: Path):
    validate_policy_snapshot(source)
    # The four accepted typed AlignedK deletions are the compiled PRODUCTION
    # policy; RuntimeK PacketStaged and legacy experimental baselines are gone.
    dropped_aligned = read_prune_groups(source / 'benchmarks/fp8_accepted_pruning.csv')
    matched = set()
    root = source / 'Utilities/TensorOperations/Cub'
    specs = {
        'narrow': ('CubReductionNarrowFlatRCooperative.cu',
                   '}  // namespace\n\nvoid launchFp8DeepRSmallKRetainedFirstStage'),
        'rotated': ('CubReductionAwkwardRotatedSharded.cu',
                    '}  // namespace\n\nnamespace {'),
        'aligned': ('CubReductionKParallelShardedTiled.cu',
                    'template <typename InputTransformT,\n          typename ReductionOpT,\n          typename OutputFinalizeT,\n          TiledRKStageTopology Topology>\nvoid dispatchInputType('),
    }
    source_text = {}
    for name, (filename, marker) in specs.items():
        text = (root / filename).read_text()
        prefix = cut_unique(text, marker, filename)
        source_text[name] = prefix

    refs = defaultdict(list)
    manifest = []
    def add(tu: str, mode: str, dtype: str, kernel: str, args: str, config: str):
        refs[tu].append(f'    out[{len(refs[tu])}] = reinterpret_cast<const void*>(&{kernel}<{args}>);')
        manifest.append({'tu':tu, 'mode':mode, 'dtype':dtype, 'kernel':kernel,
                         'config':config, 'args':args})

    for dtype, short in DTYPES:
        # OneWindow: the bucketed units. Explicitly name all template axes so we do not silently fall back to
        # FixedOp=0 (SUM).
        for o, packet, bucket, la in ONE_WINDOW_UNITS:
            for operation, label in ONE_WINDOW_OPERATIONS:
                add('narrow', 'OneWindow', short, 'fp8OneWindowRuntimeKCompleteKernel',
                    f'{dtype}, {o}, {packet}, {bucket}, {str(la).lower()}, {operation}',
                    f'O{o}/P{packet}/c{bucket}{"/la" if la else ""}/{label}')
        # Retained DeepRSmallK: P16 × 7 ownership variants × 3 compiled
        # transforms × 4 compiled FP32 combines. The five row depths are a
        # runtime operand since P5-C, except for the one-component kernel,
        # which is compiled at each depth. FP8 formats are instantiated separately.
        # Staged output is always unfinalized FP32.
        for transform, label in (('IdentityFp32', 'Identity'),
                                 ('AbsoluteValueFp32', 'ABS'),
                                 ('SquareFp32', 'Square')):
            for combine, combine_label in (('cuda::std::plus<float>', 'SUM'),
                                           ('cuda::std::multiplies<float>', 'PRODUCT'),
                                           ('PropagatingMinimumFp32', 'MIN'),
                                           ('PropagatingMaximumFp32', 'MAX')):
                for lanes in (8, 4):
                    add('narrow', 'DeepRSmallK', short,
                        'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel',
                        f'{dtype}, {combine}, {transform}, 16, {lanes}',
                        f'P16/tiny-lanes{lanes}/{label}/{combine_label}')
                # One component per lane (K 9..32) keeps a compiled depth (log2) at every
                # retained depth; the runtime depth cost it up to 4.8% (1080).
                for rows in DEEP_ROWS:
                    add('narrow', 'DeepRSmallK', short,
                        'narrowFp8PackedFlatRCooperativeFirstStageKernel',
                        f'{dtype}, {combine}, {transform}, 16, 1, {rows.bit_length() - 1}',
                        f'rows{rows}/P16/components1/{label}/{combine_label}')
                for components in (2, 4, 5, 8):
                    add('narrow', 'DeepRSmallK', short,
                        'narrowFp8PackedFlatRCooperativeFirstStageKernel',
                        f'{dtype}, {combine}, {transform}, 16, {components}',
                        f'P16/components{components}/{label}/{combine_label}')
        # AlignedK retains its independently selected mixed operation policy.
        for transform, label in (('IdentityFp32', 'Identity'),
                                 ('AbsoluteValueFp32', 'ABS'),
                                 ('SquareFp32', 'Square')):
            for topology, combos in [('Complete',ALIGNED_COMPLETE),('Staged',ALIGNED_STAGED)]:
                for p,threads in combos:
                    if topology == 'Staged' and transform == 'IdentityFp32':
                        add('aligned','AlignedK',short,'kParallelFp8StagedHoistedCombineKernel',
                            f'{dtype}, {p}, {threads}', f'{topology}/P{p}/CTA{threads}/{label}/4-ops')
                    elif topology == 'Complete' and transform == 'IdentityFp32':
                        if p == 8 and threads == 64:
                            # Typed additive SUM/MEAN is the only retained typed
                            # Identity Complete operation at these physical axes.
                            add('aligned','AlignedK',short,'kParallelFp8PacketKernel',
                                f'{dtype}, IdentityFp32, cuda::std::plus<float>, AdditiveFinalizeFp32, 8, 64, TiledRKStageTopology::Complete',
                                'Complete/P8/CTA64/Identity/typed-SUM-MEAN')
                        policy = 'Joined' if p == 2 else 'InBranch'
                        add('aligned','AlignedK',short,'kParallelFp8CompleteUnifiedKernel',
                            f'{dtype}, {p}, {threads}, Fp8CompleteFinalizePolicy::{policy}',
                            f'Complete/P{p}/CTA{threads}/Identity/{policy}-runtime-op')
                    else:
                        suffix = 'abs' if label == 'ABS' else 'square'
                        group = f'p{p}_cta{threads}_{topology.lower()}'
                        prune_key = (f'fp8_{short.lower()}', f'sum_{suffix}', group)
                        if prune_key in dropped_aligned:
                            matched.add(prune_key)
                            continue
                        add('aligned','AlignedK',short,'kParallelFp8PacketKernel',
                            f'{dtype}, {transform}, cuda::std::plus<float>, AdditiveFinalizeFp32, {p}, {threads}, TiledRKStageTopology::{topology}',
                            f'{topology}/P{p}/CTA{threads}/{label}')
        # After enabling ABS/square with PRODUCT/MIN/MAX in production AlignedK,
        # these are individually compiled typed kernel specializations, unlike
        # the shared Identity Hoisted/Unified operation implementations above.
        for transform, label, suffix in (('AbsoluteValueFp32', 'ABS', 'abs'),
                                         ('SquareFp32', 'Square', 'square')):
            for operation, op_label, reduction in ALIGNED_TYPED_OPERATIONS:
                opname = op_label.lower()
                for topology, combos in [('Complete', ALIGNED_COMPLETE), ('Staged', ALIGNED_STAGED)]:
                    for packet, threads in combos:
                        group = f'p{packet}_cta{threads}_{topology.lower()}'
                        prune_key = (f'fp8_{short.lower()}', f'{opname}_{suffix}', group)
                        if prune_key in dropped_aligned:
                            matched.add(prune_key)
                            continue
                        add('aligned', 'AlignedK', short, 'kParallelFp8PacketKernel',
                            f'{dtype}, {transform}, {reduction}, IdentityFp32, {packet}, {threads}, TiledRKStageTopology::{topology}',
                            f'{topology}/P{packet}/CTA{threads}/{label}/{op_label}')
        # Transformed additive SUM kernels are already represented in the
        # legacy 56 baseline symbols above. Omit them only if the analyzer
        # explicitly nominated that exact typed symbol.
        # RuntimeK retains all four compile-time FP32 combines. Packet kernels
        # specialize FixedOp (the established typed A/B implementation); Rotated
        # kernels use their existing canonical typed reduction functors. Both
        # FP8 formats and all three transforms are separate compiled variants.
        # Packet warp/tile/shard geometry remains runtime, and Rotated retains
        # only the six selected compiled row depths for each layout.
        for operation, op_label, combine in RUNTIME_K_OPERATIONS:
            folded = op_label in ('MIN', 'MAX')
            for transform, label in ((0, 'Identity'), (1, 'ABS'), (2, 'Square')):
                add('narrow', 'RuntimeK', short, 'fp8GeneralRParallelKTileP16CompleteKernel',
                    f'{dtype}, {transform}, {operation}',
                    f'PacketP16/Complete/{label}/{op_label}')
            for exact in (False, True):
                if folded and not exact and op_label == 'MAX':
                    continue  # PrefixReserved MAX launches the folded MIN kernel
                for rows in ROTATED_ROWS:
                    for transform, label in ((('IdentityFp32', 'Identity'),) if folded else
                                             (('IdentityFp32', 'Identity'),
                                              ('AbsoluteValueFp32', 'ABS'),
                                              ('SquareFp32', 'Square'))):
                        add('rotated', 'RuntimeK', short, 'awkwardAlignmentRotatedShardedFirstStageKernel',
                            f'{dtype}, {combine}, {transform}, {rows}, 16, 16, {str(exact).lower()}',
                            f'Rotated/{"Exact" if exact else "Prefix"}/P16/rows{rows}/{label}/'
                            f'{"MIN-MAX" if folded and not exact else op_label}')

    unknown = dropped_aligned - matched
    if unknown:
        raise ValueError(f'Accepted pruning contains unrecognized physical kernels: {sorted(unknown)}')
    # The first two TU prefixes already opened both namespaces. The aligned
    # prefix also has THOR_CUB_ENABLE_FP8_TYPES #if closed before its cut.
    out.mkdir(parents=True,exist_ok=True)
    for tu, prefix in source_text.items():
        body = '\n'.join(refs[tu]); assert refs[tu]
        end = (f'\n// Exact selected GPU symbol roots: no oracle dispatch is referenced here.\n'
               f'extern "C" void thor_fp8_selected_{tu}_kernel_references(const void** out) {{\n'
               f'{body}\n}}\n'
               f'}} // anonymous namespace\n}} // ThorImplementation::CubReductionInternal\n')
        generated = ('// GENERATED: isolated FP8 inventory size probe; do not edit or use for production.\n'
                     f'// Extracted from {specs[tu][0]}, preserving the source kernel definitions.\n'
                     + prefix + '\n#if !THOR_CUB_ENABLE_FP8_TYPES\n#error FP8 must be enabled for the isolated size probe\n#endif\n' + end)
        (out/f'{tu}.cu').write_text(generated)
    (out/'expected.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for tu in ('narrow','aligned','rotated'):
        assert (out/f'{tu}.cu').exists()
    modes = Counter(m['mode'] for m in manifest)
    expected_counts = Counter(EXPECTED_COUNTS)
    expected_counts['AlignedK'] -= len(dropped_aligned)
    assert modes == expected_counts, (modes, expected_counts)
    ow_ops = Counter((m['dtype'], m['config'].rsplit('/', 1)[-1])
                     for m in manifest if m['mode'] == 'OneWindow')
    assert ow_ops == Counter({(dtype, label): len(ONE_WINDOW_UNITS)
                             for _, dtype in DTYPES
                             for _, label in ONE_WINDOW_OPERATIONS}), ow_ops
    # Per input format: SUM/PRODUCT three Packet + 36 Rotated; MIN/MAX three Packet + six Exact each, and six
    # folded PrefixReserved MIN-MAX kernels;
    # fail if any transform, layout, or row-depth specialization disappears.
    rt_ops = Counter((m['dtype'], m['config'].rsplit('/', 1)[-1])
                     for m in manifest if m['mode'] == 'RuntimeK')
    assert rt_ops == Counter({(dtype, label): count
                             for _, dtype in DTYPES
                             for label, count in (('SUM', 39), ('PRODUCT', 39), ('MIN', 9),
                                                  ('MAX', 9), ('MIN-MAX', 6))}), rt_ops
    rt_topologies = Counter((m['dtype'], m['config'].split('/')[0])
                            for m in manifest if m['mode'] == 'RuntimeK')
    assert rt_topologies == Counter({(dtype, topology): expected
                                    for _, dtype in DTYPES
                                    for topology, expected in (
                                        ('PacketP16', 12),
                                        ('Rotated', 90))}), rt_topologies
    print('Generated 3 CUDA source TUs in',out)
    print('Expected selected first-stage specializations:',dict(modes))


def run(cmd, cwd=None, input_text=None):
    p = subprocess.run(cmd,cwd=cwd,input=input_text,capture_output=True,text=True,errors='replace')
    if p.returncode:
        raise RuntimeError(f'Failed {cmd}: {p.stdout[-2000:]} {p.stderr[-3000:]}')
    return p.stdout


def parse_func_symbols(raw):
    seen=set(); out=[]
    for line in raw.splitlines():
        parts=line.strip().split(None,7)
        if len(parts)!=8 or not parts[0].endswith(':') or parts[3]!='FUNC' or parts[6] in ('UND','ABS'):
            continue
        # readelf prints sizes above 99999 in hex (`0x249f0`); int(..., 0) reads both forms.
        try: size=int(parts[2], 0)
        except ValueError: continue
        if not size: continue
        # Some NVCC device ELFs print an extended section-index suffix
        # before the symbol (for example, `10]  9583 _ZN...`).  `split(None,
        # 7)` preserves this prefix in parts[7].  The ELF linker identity
        # itself is the final whitespace-delimited token: never send the
        # readelf prefix to c++filt or use it as an exact specialization key.
        name=parts[7].split()[-1].split('@@',1)[0]
        key=(name,parts[6],parts[1],size)
        if name and key not in seen:
            seen.add(key);out.append((name,size))
    return out


def analyze(objects: list[Path], archive: Path, out: Path):
    for name in ('cuobjdump','readelf','c++filt'):
        if not shutil.which(name):
            raise SystemExit(f'Missing {name}; requires NVIDIA toolkit/binutils')
    out.mkdir(parents=True,exist_ok=True)
    known_kernels = first_party_kernel_names(objects, archive)
    unselected_thor_kernels = known_kernels - set(FAMILIES)
    object_rows=[]; rows=[]; extra=[]; support=[]
    for object_path in objects:
        obj=object_path.resolve(strict=True)
        tu=next((n for n in ('narrow','aligned','rotated') if obj.name == f'{n}.cu.o'),None)
        if tu is None: raise SystemExit(f'Unrecognized generated object: {obj}')
        obj_out=out/'cubins'/tu
        obj_out.mkdir(parents=True,exist_ok=True)
        listing=run(['cuobjdump','-lelf',str(obj)])
        (out/f'{tu}-embedded-elf.txt').write_text(listing)
        extraction=run(['cuobjdump','-xelf','all',str(obj)],cwd=obj_out)
        (out/f'{tu}-extract-log.txt').write_text(extraction)
        cubins=list(obj_out.rglob('*.cubin'))
        if not cubins: raise SystemExit(f'No CUDA cubins in generated object: {obj}')
        object_rows.append({'object':tu,'object_bytes':obj.stat().st_size,
                            'cubin_bytes':sum(f.stat().st_size for f in cubins)})
        for cubin in cubins:
            symbols=parse_func_symbols(run(['readelf','-W','-s',str(cubin)]))
            demangled=run(['c++filt','-n'],input_text='\n'.join(s[0] for s in symbols)+'\n').splitlines()
            if len(demangled) != len(symbols):
                raise RuntimeError(f'Isolated inventory symbol/demangle mismatch in {cubin}: '
                                   f'{len(symbols)} symbols, {len(demangled)} names')
            for (mangled,size),name in zip(symbols,demangled):
                k=next((kernel for kernel in FAMILIES if is_named_kernel_function(name, kernel)),None)
                unselected=next((kernel for kernel in unselected_thor_kernels
                                 if is_named_kernel_function(name, kernel)),None) if not k else None
                kernel_like=not k and not unselected and bool(re.search(r'[Kk]ernel\s*<', name))
                # Be conservative about new, unenumerated Thor kernels from
                # included headers, not just the extracted source files.
                other_thor=kernel_like and 'ThorImplementation::' in name
                external=kernel_like and not other_thor
                mode=(FAMILIES[k] if k else 'unselected-Thor-kernel' if unselected or other_thor
                      else 'non-Thor-kernel-like-FUNC' if external else 'device-helper')
                info={'object':tu,'cubin':cubin.name,'name':name,'mangled_name':mangled,'bytes':size,'mode':mode}
                rows.append(info)
                if unselected or other_thor: extra.append(info)
                if external: support.append(info)
    # CUDA ABI functions can be present in cubins even when only the selected
    # kernels are referenced. Count helpers separately; don't pretend they're
    # identifiable per kernel.
    manifest_path = archive.parent / 'expected.json'
    if not manifest_path.exists():
        raise SystemExit(f'Missing selected-kernel manifest: {manifest_path}')
    exp = dict(Counter(item['mode'] for item in json.loads(manifest_path.read_text())))
    # Compare only first-party selected kernels against the manifest. CUDA/CCCL
    # support symbols are accounted for separately, and unexpected Thor kernels
    # are rejected through `extra` below; neither belongs in these counts.
    counts=Counter(r['mode'] for r in rows if r['mode'] in exp)
    # Multiple archs produce one symbol per architecture, so gate per-cubin arch
    # before claiming exact counts. Target should normally be sm_120 only.
    arches=set()
    for x in rows:
        m=re.search(r'sm[_-](\d+)',x['cubin'])
        arches.add('sm_'+m.group(1) if m else 'unknown')
    with (out/'device-functions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['object','cubin','mode','bytes','name','mangled_name']);w.writeheader();w.writerows(rows)
    with (out/'object-sizes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['object','object_bytes','cubin_bytes']);w.writeheader();w.writerows(object_rows)
    totals=defaultdict(int)
    for x in rows:totals[x['mode']]+=x['bytes']
    lines=[f'Isolated archive: {archive} ({archive.stat().st_size:,} bytes)',
           'Selected first-stage kernel specializations only; shared FP32 continuation is excluded.',
           f'GPU architectures: {", ".join(sorted(arches))}',
           f'Compiled CUDA objects: {len(object_rows)}',
           f'Extracted cubins: {sum(1 for _ in (out/"cubins").rglob("*.cubin"))}', '']
    for key in ('OneWindow','DeepRSmallK','AlignedK','RuntimeK'):
        lines.append(f'{key:14s} {counts[key]:5d} kernel FUNC symbols; {totals[key]:>12,} SASS text bytes')
    for key in EXPERIMENTAL_FAMILIES:
        lines.append(f'{key} (experimental, excluded from the total) {counts[key]:5d} kernel FUNC symbols; '
                     f'{totals[key]:>12,} SASS text bytes')
    lines.append(f'Non-Thor CUDA/CCCL kernel-like FUNC text: {len(support)} functions, '
                 f'{totals["non-Thor-kernel-like-FUNC"]:,} bytes')
    lines.append(f'Device helper FUNC text (unattributed): {totals["device-helper"]:,} bytes')
    lines.append(f'Combined selected first-stage SASS text: '
                 f'{sum(totals[m] for m in exp if m not in EXPERIMENTAL_FAMILIES):,} bytes')
    lines.extend(['', 'Artifact bytes, with metadata and (possibly) compression:'])
    for x in object_rows:
        lines.append(f'  {x["object"]}: object {x["object_bytes"]:,}; extracted cubins {x["cubin_bytes"]:,}')
    lines.extend(['','NOTE: Archive bytes and extracted cubin bytes are NOT incremental libThor.so size.',
                  'This probe contains first-stage kernels only, not the reused production FP32 continuation.',
                  'OneWindow is bucketed by components per lane (runtime K); RuntimeK warp and tile decisions are runtime.',
                  'Inspect device-functions.csv for exact symbol sizes.'])
    bad = bool(extra) or (len(arches)==1 and counts != Counter(exp))
    if support:
        lines.append(f'NOTE: {len(support)} non-Thor kernel-like device functions are present '
                     'but not counted as selected Thor kernels; see device-functions.csv:')
        for item in support:
            lines.append(f'  {item["object"]}: {item["bytes"]} bytes: {item["name"]}')
    if extra:
        lines.append(f'ERROR: {len(extra)} compiled Thor __global__ kernels are outside the selected inventory:')
        for item in extra:
            lines.append(f'  {item["object"]}: {item["bytes"]} bytes: {item["name"]}')
    if len(arches)==1 and counts != Counter(exp):
        lines.append(f'ERROR expected {exp}, got {dict(counts)}')
        manifest = json.loads(manifest_path.read_text())
        for mode in sorted(m for m in exp if counts[m] != exp[m]):
            compiled = set()
            for r in rows:
                if r['mode'] != mode:
                    continue
                for kernel in (k for k, fam in FAMILIES.items() if fam == mode):
                    m = re.search(re.escape(kernel) + r'\s*<(.+?)>\s*\(', r['name'])
                    if m:
                        compiled.add(specialization_key(kernel, m.group(1)))
            missing = [e for e in manifest if e['mode'] == mode
                       and specialization_key(e['kernel'], e['args']) not in compiled]
            lines.append(f'  {mode}: {len(missing)} manifest specializations without a matching kernel symbol'
                         ' (first 20):')
            lines.extend(f'    {e["dtype"]} {e["config"]}: {e["kernel"]}<{e["args"]}>' for e in missing[:20])
            helpers = [r for r in rows if r['mode'] == 'device-helper'
                       and any(k in r['name'] for k, fam in FAMILIES.items() if fam == mode)]
            lines.append(f'  {mode}: {len(helpers)} device-helper symbols naming this family (first 10):')
            lines.extend(f'    {r["bytes"]} bytes: {r["name"][:200]}' for r in helpers[:10])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if bad:raise SystemExit('Isolated inventory membership assertion failed: inspect device-functions.csv')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    g=sub.add_parser('generate');g.add_argument('--source',type=Path,required=True);g.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('analyze');a.add_argument('--archive',type=Path,required=True)
    a.add_argument('--objects',type=Path,nargs='+',required=True);a.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.action=='generate':generate(args.source,args.out)
    else:analyze(args.objects,args.archive,args.out)

if __name__=='__main__':main()
