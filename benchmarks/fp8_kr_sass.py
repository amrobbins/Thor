#!/usr/bin/env python3
"""SASS capture of the retained FP8 R / KR kernels (1149). Read-only; standard library + CUDA toolkit binaries.

Extracts every device ELF from the built CubReductionNarrowFlatRCooperative.cu.o, and for each fp8KR* kernel writes
  <out>/<arch>/<kernel>.sass   the full SASS listing (cuobjdump -sass), and
  <out>/kr_sass_summary.csv    resources (registers, stack, local / spill, shared) and an instruction mix:
                               total instructions, global loads by width, global stores, shuffles, conversions,
                               FP32 / integer ALU, 64-bit address math, branches / reconvergence, local (spill)
                               memory traffic.
and prints the summary. Zip <out> and attach it. Requires cuobjdump and c++filt on PATH.

Usage: fp8_kr_sass.py --build BUILD_DIR --out OUT_DIR [--filter fp8KR] [--object NAME.cu.o]

1189: --object picks another translation unit's object (default the NarrowFlat one), e.g.
CubReductionKParallelShardedTiled.cu.o for the AlignedK kernels; FP32 kernel names (fp32...Kernel) are shortened like
FP8 ones.
"""

import argparse
import hashlib
import collections
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OBJECT_NAME = "CubReductionNarrowFlatRCooperative.cu.o"
RESOURCE = re.compile(r"Function\s+(\S+):\s*$")
RESOURCE_VALUES = re.compile(r"(REG|STACK|SHARED|LOCAL):(\d+)")
SASS_FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
SASS_INSTRUCTION = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]\s+)?([A-Z][A-Z0-9_.]*)")

# Instruction-mix columns: name -> predicate on the full opcode (e.g. "LDG.E.128.CONSTANT").
MIX = {
    "ldg128": lambda op: op.startswith("LDG") and ".128" in op,
    "ldg64": lambda op: op.startswith("LDG") and ".64" in op,
    "ldg32": lambda op: op.startswith("LDG") and not any(w in op for w in (".128", ".64", ".U8", ".S8", ".U16", ".S16")),
    "ldg8_16": lambda op: op.startswith("LDG") and any(w in op for w in (".U8", ".S8", ".U16", ".S16")),
    "stg": lambda op: op.startswith("STG"),
    "lds_sts": lambda op: op.startswith(("LDS", "STS")),
    "ldl_stl": lambda op: op.startswith(("LDL", "STL")),
    "shfl": lambda op: op.startswith("SHFL"),
    "convert": lambda op: op.startswith(("F2F", "F2FP", "HADD2.F32", "I2F", "F2I", "PRMT")),
    "fp32": lambda op: op.startswith(("FADD", "FMUL", "FFMA", "FMNMX", "FSEL", "FSETP")),
    "int_alu": lambda op: op.startswith(("IADD3", "IMAD", "LOP3", "SHF", "ISETP", "SEL", "LEA", "IABS", "VIADD", "IMNMX")),
    "imad_wide": lambda op: op.startswith("IMAD.WIDE"),
    "branch": lambda op: op.startswith(("BRA", "BSSY", "BSYNC", "WARPSYNC", "BREAK", "BRX")),
    "bar": lambda op: op.startswith("BAR"),
    "atom": lambda op: op.startswith(("ATOM", "RED")),
}


def run(cmd, cwd=None, text_input=None):
    result = subprocess.run(cmd, cwd=cwd, input=text_input, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout


def demangle(names):
    if not names:
        return {}
    out = run(["c++filt"], text_input="\n".join(names) + "\n").splitlines()
    return dict(zip(names, out))


def short_name(demangled):
    # "void ns::(anonymous namespace)::fp8KRValueGroupKernel<__nv_fp8_e4m3, 0, 0>(args)" -> "fp8KRValueGroupKernel<e4m3,0,0>"
    # 1185-1186: any retained kernel (fp8KR..., fp8OneWindow..., narrowFp8Packed...), template arguments kept whole
    # (balanced <>), namespaces dropped, so --filter can select RK kernels and names stay distinct.
    # 1191: any kernel name (\w+Kernel before its template arguments or parameters), e.g. the rotated
    # awkwardAlignmentRotatedShardedFirstStageKernel, so no listing is named after the whole signature.
    match = re.search(r"\w*[Ff]p(?:8|32)\w*Kernel", demangled) or re.search(r"\w+Kernel(?=[<(])", demangled)
    if not match:
        return demangled.replace("__nv_fp8_", "").replace(" ", "")
    name, rest = match.group(0), demangled[match.end():]
    if rest.startswith("<"):
        depth = 0
        for i, ch in enumerate(rest):
            depth += ch == "<"
            depth -= ch == ">"
            if depth == 0:
                name += rest[:i + 1]
                break
    name = re.sub(r"(\w+::|\(anonymous namespace\)::)", "", name)
    return name.replace("__nv_fp8_", "").replace(" ", "")


def parse_resources(text):
    resources = {}
    current = None
    for line in text.splitlines():
        m = RESOURCE.search(line)
        if m:
            current = m.group(1)
            continue
        if current and "REG:" in line:
            resources[current] = {k.lower(): int(v) for k, v in RESOURCE_VALUES.findall(line)}
            current = None
    return resources


def parse_sass(text):
    functions = collections.OrderedDict()
    current = None
    for line in text.splitlines():
        m = SASS_FUNCTION.match(line)
        if m:
            current = m.group(1)
            functions[current] = []
            continue
        if current is not None:
            functions[current].append(line)
    return functions


def mix(lines):
    counts = collections.Counter()
    for line in lines:
        m = SASS_INSTRUCTION.match(line)
        if not m:
            continue
        op = m.group(1)
        if op == "NOP":
            continue
        counts["instructions"] += 1
        for column, predicate in MIX.items():
            if predicate(op):
                counts[column] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filter", default="fp8KR", help="substring of the demangled kernel names to keep")
    parser.add_argument("--object", default=OBJECT_NAME, help="object file name to search for under --build")
    args = parser.parse_args()
    for program in ("cuobjdump", "c++filt"):
        if shutil.which(program) is None:
            sys.exit(f"{program} not found on PATH")
    objects = sorted(args.build.rglob(args.object))
    if not objects:
        sys.exit(f"no {args.object} under {args.build} (build Thor first)")
    obj = objects[0].resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        run(["cuobjdump", "-xelf", "all", str(obj)], cwd=tmp)
        cubins = sorted(Path(tmp).glob("*.cubin")) or sorted(Path(tmp).iterdir())
        if not cubins:
            sys.exit(f"no device ELF in {obj}")
        for cubin in cubins:
            arch_match = re.search(r"sm_?(\d+)", cubin.name)
            arch = f"sm_{arch_match.group(1)}" if arch_match else cubin.stem
            resources = parse_resources(run(["cuobjdump", "-res-usage", str(cubin)]))
            functions = parse_sass(run(["cuobjdump", "-sass", str(cubin)]))
            names = demangle(list(functions))
            kept = {mangled: lines for mangled, lines in functions.items() if args.filter in names[mangled]}
            arch_dir = args.out / arch
            arch_dir.mkdir(exist_ok=True)
            for mangled, lines in kept.items():
                name = short_name(names[mangled])
                safe = re.sub(r"[^A-Za-z0-9_,.-]+", "_", name)
                if len(safe) > 200:  # 1191: stay under the 255-byte file name limit
                    safe = safe[:180] + "_" + hashlib.sha1(safe.encode()).hexdigest()[:12]
                (arch_dir / f"{safe}.sass").write_text(f"// {names[mangled]}\n// {mangled}\n" + "\n".join(lines) + "\n")
                row = {"arch": arch, "kernel": name}
                row.update({k: resources.get(mangled, {}).get(k, "") for k in ("reg", "stack", "local", "shared")})
                counts = mix(lines)
                row["instructions"] = counts["instructions"]
                row.update({column: counts[column] for column in MIX})
                rows.append(row)

    if not rows:
        sys.exit(f"no kernels matching '{args.filter}' in {obj}")
    columns = ["arch", "kernel", "reg", "stack", "local", "shared", "instructions"] + list(MIX)
    with open(args.out / "kr_sass_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"# object {obj}")
    print(f"# {len(rows)} kernels; listings in {args.out}/<arch>/, summary in {args.out}/kr_sass_summary.csv")
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.rjust(widths[c]) if c not in ("arch", "kernel") else c.ljust(widths[c]) for c in columns))
    for r in sorted(rows, key=lambda r: (r["arch"], r["kernel"])):
        print("  ".join(str(r[c]).rjust(widths[c]) if c not in ("arch", "kernel") else str(r[c]).ljust(widths[c])
                        for c in columns))


if __name__ == "__main__":
    main()
