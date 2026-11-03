"""Host-only checks for the KR training sweep summary (1151-1152)."""
import gzip
import pathlib
import tempfile
import unittest

import contextlib
import io

from benchmarks.fp8_kr_training import Cell, compare, load_cells, read, row_bucket

HEADER = ("kind,cell_id,heldout,random_geometry,K,R,input_bytes,target_mib,dtype,operation,shape,family,"
          "rows_per_thread,group_lanes,chunk_bytes,chunks,vector,blocks,threads,in_list,list_position,"
          "production,is_production,stamp_ms,scratch_bytes,launches_per_sample,input_views,output_views,"
          "abandoned,probe_us,median_us,min_us,reference_us,moved_tbps,read_tbps,sample0_us,sample1_us,sample2_us")


def row(shape, family, position, median, abandoned=0, production="g32"):
    in_list = 1 if position else 0
    return (f"value,7,0,1,1000,65536,65536000,0,fp8_e4m3,sum,{shape},{family},1,1,0,1,0,1,1,{in_list},{position},"
            f"{production},{int(shape == production)},12.0,0,1,32,1024,{abandoned},{median},{median},{median},40,"
            f"{65540000 / median / 1e6:.4f},{65536000 / median / 1e6:.4f},{median},{median * 1.01},{median * 0.99}")


def write(path, lines):
    with gzip.open(path, "wt") as stream:
        stream.write("# fp8_kr_training=1 version=1152 copy_bandwidth_tbps=1.5700\n" + HEADER + "\n")
        for line in lines:
            stream.write(line + "\n")


class TestKRTraining(unittest.TestCase):
    def test_oracle_list_and_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "t.csv.gz"
            write(path, [row("g32", "group", 1, 60.0), row("s16384", "split", 2, 50.0),
                         row("s4096", "split", 0, 44.0), row("g1", "group", 0, 900.0, abandoned=1),
                         "# unsupported,value,8,sum"])
            header, unsupported, rows = read(str(path))
            self.assertEqual(len(header), 1)
            self.assertEqual(len(unsupported), 1)
            cell = Cell(rows)
            self.assertEqual(cell.abandoned, 1)
            self.assertEqual(cell.oracle, "s4096")       # an unlisted shape is the oracle
            self.assertEqual(cell.list_best, "s16384")
            self.assertEqual(cell.at_depth(1), "g32")
            self.assertEqual(cell.at_depth(6), "s16384")
            self.assertAlmostEqual(cell.regret(cell.at_depth(1)), 60.0 / 44.0 - 1.0)
            self.assertAlmostEqual(cell.regret(cell.list_best, cell.list_best), 0.0)
            self.assertEqual(cell.family(cell.oracle), "split")
            self.assertEqual(row_bucket(1, 1 << 24), "pure R")
            self.assertEqual(row_bucket(1000, 65536), "R 8K-64K")
            self.assertEqual(cell.production_shape(), "g32")
            self.assertAlmostEqual(cell.regret(cell.production_shape(), cell.list_best), 60.0 / 50.0 - 1.0)

    def test_compare_reports_run_to_run_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            first = pathlib.Path(directory) / "a.csv.gz"
            second = pathlib.Path(directory) / "b.csv.gz"
            write(first, [row("g32", "group", 1, 60.0), row("s4096", "split", 0, 44.0)])
            write(second, [row("g32", "group", 1, 60.06), row("s4096", "split", 0, 44.0)])
            _, _, _, cells = load_cells(str(first))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                compare(cells, str(second), 16, 5)
            text = out.getvalue()
            self.assertIn("1 shared cells", text)
            self.assertIn("0.10%", text)  # the g32 pair: 60.06 / 60 - 1 (the max column)
            self.assertIn("100%", text)   # same oracle shape


if __name__ == "__main__":
    unittest.main()
