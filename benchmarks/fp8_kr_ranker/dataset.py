"""Training dataset of `thor_fp8_kr_training_sweep` (1158+). Standard library only.

One cell = one site [K, R] with one sampled semantics; every shape of production's candidate pool
(fp8KRCandidatePool) was timed with the RK sweep's method. The C++ ranker features f0..f23 (fp8KRRankerFeatures)
come straight from the sweep, so no feature is re-derived here. Rows with variant=partials (--split-stages) are not
candidates and are skipped. An abandoned candidate (its probe launch over 20x the line-rate reference) keeps its probe
time: it is slow, and the ranker should learn so.
"""

import csv
import gzip
import math
from dataclasses import dataclass, field

FEATURE_NAMES = [
    "log2_rows", "log2_row_bytes", "log2_bytes", "family", "rows_per_thread", "log2_lanes", "log2_chunk_bytes",
    "log2_chunks", "log2_blocks", "log2_blocks_per_sm", "log2_threads", "log2_bytes_per_thread",
    "log2_bytes_per_block", "log2_words_per_lane", "row_bytes_mod4", "row_bytes_mod16", "log2_rows_per_sm", "kind",
    "combine", "transform", "e5m2", "log2_output_bytes", "list_position", "log2_sms",
]
FEATURE_COUNT = len(FEATURE_NAMES)  # 24, Fp8KRSegments.h FP8_KR_RANKER_FEATURES
UNLISTED = 1000  # evaluate.list_order: listed shapes in list order, then the rest in pool order


@dataclass
class Candidate:
    family: str
    configuration: str  # the shape name (fp8KRShapeName)
    local_rank: int  # = list_position (for evaluate.order_by_scores' leader rule, unused by KR)
    list_position: int  # heuristic list position, or UNLISTED + pool position
    features: list  # FEATURE_COUNT floats (float32-exact values)
    median_ms: float  # microseconds (evaluate only takes ratios); the name keeps fp8_ranker.evaluate working
    abandoned: bool


@dataclass
class Cell:
    cell_id: int
    kind: str
    heldout: bool
    random_geometry: bool
    k: int  # rows
    r: int  # row bytes
    input_bytes: int
    dtype: str
    operation: str
    candidates: list = field(default_factory=list)

    @property
    def shape(self):
        """Geometry group for cross-validation: the sweep holds out row lengths R, so folds group by R too."""
        return self.r

    @property
    def outer(self):
        return 1

    @property
    def oracle_ms(self):
        return min(c.median_ms for c in self.candidates)

    def regrets(self):
        oracle = self.oracle_ms
        return [c.median_ms / oracle for c in self.candidates]


def _open(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load(path):
    """(header tokens, cells in cell-id order). Raises on malformed or incomplete cells."""
    header, lines = {}, []
    with _open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                for token in line[1:].split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        header.setdefault(key, value)
                continue
            lines.append(line)
    if header.get("fp8_kr_ranker_training") != "1":
        raise ValueError(f"{path}: not a KR ranker dataset (needs thor_fp8_kr_training_sweep 1158+)")
    if int(header.get("feature_count", "0")) != FEATURE_COUNT:
        raise ValueError(f"{path}: feature_count {header.get('feature_count')} != {FEATURE_COUNT}")
    cells = {}
    for row in csv.DictReader(lines):
        if row.get("variant", "full") != "full":
            continue
        cell_id = int(row["cell_id"])
        cell = cells.get(cell_id)
        if cell is None:
            cell = cells[cell_id] = Cell(cell_id, row["kind"], row["heldout"] == "1", row["random_geometry"] == "1",
                                         int(row["K"]), int(row["R"]), int(row["input_bytes"]), row["dtype"],
                                         row["operation"])
        features = [float(row[f"f{i}"]) for i in range(FEATURE_COUNT)]
        median = float(row["median_us"])
        if not all(math.isfinite(x) for x in features) or not median > 0:
            raise ValueError(f"cell {cell_id}: non-finite feature or non-positive time")
        listed = int(row["list_position"])
        position = listed if listed > 0 else UNLISTED + int(row["pool_position"])
        cell.candidates.append(Candidate(row["family"], row["shape"], position, position, features, median,
                                         row["abandoned"] == "1"))
    result = [cells[i] for i in sorted(cells)]
    if not result:
        raise ValueError(f"{path}: no cells")
    for cell in result:
        names = [c.configuration for c in cell.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"cell {cell.cell_id}: duplicate candidate shape")
    return header, result


def relevance(regret):
    """Graded label for the ranking objective (as the RK ranker): ties within noise (1%) are equally good."""
    if regret <= 1.01:
        return 4
    if regret <= 1.03:
        return 3
    if regret <= 1.07:
        return 2
    if regret <= 1.15:
        return 1
    return 0
