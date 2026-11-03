"""Training dataset of `thor_fp8_retained_cold_l2_sweep --training` (1133+). Standard library only.

One cell = one geometry (outer, R, K) with one sampled semantics; every legal retained candidate was timed end to end.
The C++ ranker features f0..f24 come straight from the sweep (rankerFeatures at the candidate's local rank), so no
feature is re-derived here. The model sees those 25 plus the aggregate kind (FEATURE_NAMES).
"""

import csv
import gzip
import math
from dataclasses import dataclass, field

KINDS = {"value": 0, "arg_index": 1, "arg_value_index": 2}
BASE_FEATURES = 25
FEATURE_NAMES = [
    "log2_outer", "log2_r", "log2_k", "log2_elements", "log2_outputs", "log2_r_over_k", "e4m3", "combine",
    "transform", "family", "local_rank", "log2_packet", "log2_threads", "log2_shards", "log2_warps", "log2_tiles",
    "log2_outputs_per_warp", "log2_rows", "mode", "log2_outer_shards", "log2_outer_shards_tiles",
    "log2_partial_bytes", "log2_input_mib", "log2_r_over_rows", "log2_k_over_lanes", "aggregate_kind",
]
FEATURE_COUNT = len(FEATURE_NAMES)  # 26


@dataclass
class Candidate:
    family: str
    configuration: str
    local_rank: int
    list_position: int
    staged: bool
    frozen_margin: float
    features: list  # FEATURE_COUNT floats (float32-exact values)
    median_ms: float
    samples: list


@dataclass
class Cell:
    cell_id: int
    kind: str
    heldout: bool
    random_geometry: bool
    outer: int
    r: int
    k: int
    target_mib: int
    dtype: str
    operation: str
    candidates: list = field(default_factory=list)

    @property
    def shape(self):
        """Geometry group for cross-validation: every working set of an (R, K) shape stays in one fold."""
        return (self.r, self.k)

    @property
    def oracle_ms(self):
        return min(c.median_ms for c in self.candidates)

    def regrets(self):
        oracle = self.oracle_ms
        return [c.median_ms / oracle for c in self.candidates]


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """Cells in file order (candidates in sweep order). Raises on malformed or incomplete cells."""
    header = {}
    cells = {}
    with _open(path) as handle:
        lines = []
        for line in handle:
            if line.startswith("#"):
                for token in line[1:].split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        header[key] = value
                continue
            lines.append(line)
    if header.get("fp8_ranker_training") != "1":
        raise ValueError(f"{path}: not a --training dataset (missing '# fp8_ranker_training=1')")
    for row in csv.DictReader(lines):
        cell_id = int(row["cell_id"])
        cell = cells.get(cell_id)
        if cell is None:
            cell = cells[cell_id] = Cell(cell_id, row["kind"], row["heldout"] == "1", row["random_geometry"] == "1",
                                         int(row["outer"]), int(row["R"]), int(row["K"]), int(row["target_mib"]),
                                         row["dtype"], row["operation"])
        if row["kind"] not in KINDS:
            raise ValueError(f"unknown kind {row['kind']!r}")
        features = [float(row[f"f{i}"]) for i in range(BASE_FEATURES)] + [float(KINDS[row["kind"]])]
        samples = [float(row[f"sample{i}_ms"]) for i in range(5)]
        median = float(row["total_median_ms"])
        if not all(math.isfinite(x) for x in features) or not median > 0:
            raise ValueError(f"cell {cell_id}: non-finite feature or non-positive time")
        cell.candidates.append(Candidate(row["family"], row["configuration"], int(row["local_rank"]),
                                         int(row["list_position"]), row["staged"] == "1",
                                         float(row["frozen_margin"]), features, median, samples))
    result = [cells[i] for i in sorted(cells)]
    if not result:
        raise ValueError(f"{path}: no cells")
    for cell in result:
        names = [(c.family, c.configuration) for c in cell.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"cell {cell.cell_id}: duplicate candidate configuration")
    return header, result


def relevance(regret):
    """Graded label for the ranking objective: ties within measurement noise (1%) are equally good."""
    if regret <= 1.01:
        return 4
    if regret <= 1.03:
        return 3
    if regret <= 1.07:
        return 2
    if regret <= 1.15:
        return 1
    return 0
