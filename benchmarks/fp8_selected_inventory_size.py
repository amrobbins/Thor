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
K_VALUES = (1,2,3,4,5,7,8,9,11,13,15,16,17,19,31,32,63,64,127,128,129,255,256,497,512)
ONE_WINDOW = ((1,1,16),(1,2,16),(1,4,16),(1,8,16),(2,1,16),(4,1,16),(8,1,16),
              (8,1,4),(4,1,8),(8,1,8))
DEEP_ROWS = (64,128,256,512,1024)
ROTATED_ROWS = (32,64,128,256,512,1024)
ALIGNED_STAGED = ((2,32),(2,64),(2,128),(8,32),(8,64),(8,128))
ALIGNED_COMPLETE = ((2,32),(8,32),(8,64))

FAMILIES = {
    'fp8GeneralRParallelPackedCompleteKernel': 'OneWindow',
    'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel': 'DeepRSmallK',
    'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'DeepRSmallK',
    'kParallelFp8PacketKernel': 'AlignedK',
    'fp8GeneralRParallelKTileP16CompleteKernel': 'RuntimeK',
    'fp8GeneralRParallelKTileP16FirstStageKernel': 'RuntimeK',
    'awkwardAlignmentRotatedShardedFirstStageKernel': 'RuntimeK',
}


# Names of first-party __global__ kernels are extracted from the same build-local
# source TUs we compile, rather than guessed from a broad "Kernel<" regex.
# This distinguishes an accidentally instantiated Thor kernel from CUDA/CCCL
# support kernels whose names legitimately contain "kernel<".
CUDA_GLOBAL_DECLARATION = re.compile(
    r'__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\('
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
    # include an anonymous namespace qualifier as well.
    return bool(re.search(r'(?:::|\b)' + re.escape(kernel_name) + r'\s*[<(]', name))


def cut_unique(text: str, marker: str, source: str) -> str:
    count = text.count(marker)
    if count != 1:
        raise ValueError(f'{source}: expected one {marker!r}, got {count}. Refusing source extraction.')
    return text.split(marker, 1)[0]




def validate_policy_snapshot(source: Path):
    """Fail closed if the inventory literals drift from the policy headers."""
    b = source / 'benchmarks'
    ow = (b / 'Fp8OneWindowSelector.h').read_text()
    marker = 'inline constexpr std::array<Variant, 10> candidates = '
    if ow.count(marker) != 1:
        raise ValueError('OneWindow inventory declaration changed')
    variant_line = ow.split(marker,1)[1].split('}}};',1)[0]
    entries = tuple(tuple(map(int, a)) for a in re.findall(r'(\d+),(\d+),(\d+)',variant_line))
    if entries != ONE_WINDOW:
        raise ValueError('OneWindow variants changed')
    kline = ow.split('constexpr bool compiledK(uint64_t k)',1)[1].split('default:',1)[0]
    if tuple(int(x) for x in re.findall(r'case (\d+):',kline)) != K_VALUES:
        raise ValueError('OneWindow compiled K changed')
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


def generate(source: Path, out: Path):
    validate_policy_snapshot(source)
    root = source / 'Utilities/TensorOperations/Cub'
    specs = {
        'narrow': ('CubReductionNarrowFlatRCooperative.cu',
                   '}  // namespace\n\nvoid launchFp8NarrowLowPrecisionFlatRCooperativeCalibrationFirstStage'),
        'rotated': ('CubReductionAwkwardRotatedSharded.cu',
                    '}  // namespace\n\nnamespace {'),
        'aligned': ('CubReductionKParallelShardedTiled.cu',
                    'template <typename InputTransformT,\n          typename ReductionOpT,\n          typename OutputFinalizeT,\n          TiledRKStageTopology Topology>\nvoid dispatchInputType('),
    }
    source_text = {}
    for name, (filename, marker) in specs.items():
        text = (root / filename).read_text()
        prefix = cut_unique(text, marker, filename)
        if name == 'narrow':
            # Discard the one non-template calibration-only global kernel in the prefix.
            roofline_start = '__global__ void fp8P4ReadWriteRooflineKernel'
            roofline_end = 'template <typename InputT, int K, int OutputsPerWarp>\nvoid launchFp8CompactMultiOutputP4'
            assert prefix.count(roofline_start) == 1 and prefix.count(roofline_end) == 1
            start, end = prefix.index(roofline_start), prefix.index(roofline_end)
            prefix = prefix[:start] + prefix[end:]
            # Its non-template occupancy query must also go.
            occupancy_start = '[[nodiscard]] CubKernelOccupancyInfo queryFp8P4ReadWriteRooflineOccupancy() {'
            occupancy_end = 'template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowsPerShard, int PacketBytes, int MaxComponentsPerLane>'
            assert prefix.count(occupancy_start) == 1
            start = prefix.index(occupancy_start)
            end = prefix.index(occupancy_end, start)
            prefix = prefix[:start] + prefix[end:]
        source_text[name] = prefix

    refs = defaultdict(list)
    manifest = []
    def add(tu: str, mode: str, dtype: str, kernel: str, args: str, config: str):
        refs[tu].append(f'    out[{len(refs[tu])}] = reinterpret_cast<const void*>(&{kernel}<{args}>);')
        manifest.append({'tu':tu, 'mode':mode, 'dtype':dtype, 'kernel':kernel,
                         'config':config, 'args':args})

    for dtype, short in DTYPES:
        ow_emitted = set()
        for k in K_VALUES:
            for o,w,p in ONE_WINDOW:
                lanes=32//o
                cap=lanes*p if k%p==0 else lanes*p-(p-1)
                if k <= cap and (k,p,o) not in ow_emitted:
                    ow_emitted.add((k,p,o))
                    add('narrow','OneWindow',short,'fp8GeneralRParallelPackedCompleteKernel',
                        f'{dtype}, {k}, {p}, {o}, false',f'K{k}/O{o}/P{p}')
        for rows in DEEP_ROWS:
            for lanes in (8,4):
                add('narrow','DeepRSmallK',short,'narrowFp8PackedFlatRCooperativeTinyKFirstStageKernel',
                    f'{dtype}, cuda::std::plus<float>, IdentityFp32, {rows}, 16, {lanes}',
                    f'rows{rows}/P16/tiny-lanes{lanes}')
            for components in (1,2,4,5,8):
                add('narrow','DeepRSmallK',short,'narrowFp8PackedFlatRCooperativeFirstStageKernel',
                    f'{dtype}, cuda::std::plus<float>, IdentityFp32, {rows}, 16, {components}',
                    f'rows{rows}/P16/components{components}')
        for topology, combos in [('Complete',ALIGNED_COMPLETE),('Staged',ALIGNED_STAGED)]:
            for p,threads in combos:
                add('aligned','AlignedK',short,'kParallelFp8PacketKernel',
                    f'{dtype}, IdentityFp32, cuda::std::plus<float>, AdditiveFinalizeFp32, {p}, {threads}, TiledRKStageTopology::{topology}',
                    f'{topology}/P{p}/CTA{threads}')
        for name in ('fp8GeneralRParallelKTileP16CompleteKernel', 'fp8GeneralRParallelKTileP16FirstStageKernel'):
            add('narrow','RuntimeK',short,name,dtype,'PacketP16/'+('Complete' if 'Complete' in name else 'Staged'))
        for exact in (False,True):
            for rows in ROTATED_ROWS:
                add('rotated','RuntimeK',short,'awkwardAlignmentRotatedShardedFirstStageKernel',
                    f'{dtype}, cuda::std::plus<float>, IdentityFp32, {rows}, 16, 16, {str(exact).lower()}',
                    f'Rotated/{"Exact" if exact else "Prefix"}/P16/rows{rows}')

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
    assert modes == Counter({'OneWindow':252,'DeepRSmallK':70,'AlignedK':18,'RuntimeK':28}), modes
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
        try: size=int(parts[2])
        except ValueError: continue
        if not size: continue
        name=parts[7].split('@@',1)[0]
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
                info={'object':tu,'cubin':cubin.name,'name':name,'bytes':size,'mode':mode}
                rows.append(info)
                if unselected or other_thor: extra.append(info)
                if external: support.append(info)
    # CUDA ABI functions can be present in cubins even when only the selected
    # kernels are referenced. Count helpers separately; don't pretend they're
    # identifiable per kernel.
    exp={'OneWindow':252,'DeepRSmallK':70,'AlignedK':18,'RuntimeK':28}
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
        w=csv.DictWriter(f,fieldnames=['object','cubin','mode','bytes','name']);w.writeheader();w.writerows(rows)
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
    lines.append(f'Non-Thor CUDA/CCCL kernel-like FUNC text: {len(support)} functions, '
                 f'{totals["non-Thor-kernel-like-FUNC"]:,} bytes')
    lines.append(f'Device helper FUNC text (unattributed): {totals["device-helper"]:,} bytes')
    lines.append(f'Combined selected first-stage SASS text: {sum(totals[m] for m in exp):,} bytes')
    lines.extend(['', 'Artifact bytes, with metadata and (possibly) compression:'])
    for x in object_rows:
        lines.append(f'  {x["object"]}: object {x["object_bytes"]:,}; extracted cubins {x["cubin_bytes"]:,}')
    lines.extend(['','NOTE: Archive bytes and extracted cubin bytes are NOT incremental libThor.so size.',
                  'This probe contains first-stage kernels only, not the reused production FP32 continuation.',
                  'OneWindow K templates are specialized; RuntimeK warp and tile decisions are runtime.',
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
    if len(arches)==1 and counts != Counter(exp):lines.append(f'ERROR expected {exp}, got {dict(counts)}')
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
