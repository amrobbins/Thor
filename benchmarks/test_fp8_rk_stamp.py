"""Tests of fp8_rk_stamp.py on a tiny synthetic --training --stamp output."""
import io
import contextlib
import os
import tempfile
import unittest

import fp8_rk_stamp

FEATURES = ",".join(f"f{i}" for i in range(25))
HEADER = ("kind,cell_id,heldout,random_geometry,outer,R,K,target_mib,dtype,operation,family,configuration,local_rank,"
          "list_position,staged,frozen_margin," + FEATURES + ",total_median_ms,sample0_ms,sample1_ms,sample2_ms,"
          "sample3_ms,sample4_ms")


def row(kind, cell, outer, r, k, config, margin, ms):
    zeros = ",".join("0" for _ in range(25))
    return f"{kind},{cell},0,0,{outer},{r},{k},1,fp8_e4m3,sum,f,{config},1,1,0,{margin},{zeros},{ms},{ms},{ms},{ms},{ms},{ms}"


class StampTest(unittest.TestCase):
    def setUp(self):
        lines = ["# fp8_ranker_training=1 version=1133", "#stamp,cell_id,stamp_ms,candidates,winner", HEADER,
                 row("value", 0, 64, 16, 64, "a", 3.0, 0.003),  # 64 KiB: a 3 us, b 4 us, slow 300 us
                 row("value", 0, 64, 16, 64, "b", 2.0, 0.004),
                 row("value", 0, 64, 16, 64, "slow", 1.0, 0.300),
                 "#stamp,0,11.5,3,b",
                 row("arg_index", 1, 4096, 64, 1024, "x", 1.0, 0.200),  # 256 MiB
                 row("arg_index", 1, 4096, 64, 1024, "y", 2.0, 0.190),
                 "#stamp,1,30.0,2,y"]
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        handle.write("\n".join(lines) + "\n")
        handle.close()
        self.path = handle.name

    def tearDown(self):
        os.unlink(self.path)

    def test_load_and_report(self):
        cells, stamps = fp8_rk_stamp.load(self.path)
        self.assertEqual(stamps[0], (11.5, 3, "b"))
        self.assertAlmostEqual(cells[0]["times"]["slow"], 300.0)
        self.assertEqual(cells[1]["bytes"], 4096 * 64 * 1024)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(fp8_rk_stamp.main([self.path]), 0)
        text = out.getvalue()
        self.assertIn("33.3%", text)  # cell 0 picked b (4 us) over a (3 us)
        self.assertIn("ARG", text)

    def test_contest_estimate(self):
        before, after = fp8_rk_stamp.contest_us([3.0, 4.0, 300.0], 3)
        self.assertAlmostEqual(before, (3 + 4 + 300) * 8)
        # slow is abandoned (300 > 3 x 8); a and b run 32 launches x 7 rounds; every one has warmup + probe.
        self.assertAlmostEqual(after, 2 * (3 + 4 + 300) + 7 * 32 * (3 + 4))


if __name__ == "__main__":
    unittest.main()
