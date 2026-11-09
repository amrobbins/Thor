"""Tests of ragged_reduction_audit.py on a tiny synthetic thor_ragged_reduction_audit output."""
import argparse
import contextlib
import io
import os
import tempfile
import unittest

import ragged_reduction_audit as audit

HEADER = ("cell_id,shape_id,distribution,kind,operation,dtype,offset_dtype,D,rows,capacity_rows,segments,mean_len,"
          "max_len,empty_segments,input_bytes,offset_bytes,output_bytes,moved_bytes,stamp_ms,passes,pass_count,"
          "probe_us,launches_per_sample,sample0_us,sample1_us,sample2_us,sample3_us,sample4_us,median_us,copy_us,tbps,"
          "of_copy,verified")


def row(cell, shape, distribution, kind, d, rows, segments, us, passes="vector_narrow_rl8", max_len=None, verified=1):
    input_bytes = 4 * rows * d
    offset_bytes = 4 * (segments + 1)
    output_bytes = 4 * segments * d
    moved = input_bytes + offset_bytes + output_bytes
    samples = ",".join(str(us * f) for f in (0.99, 1.0, 1.0, 1.01, 1.02))
    max_len = rows // segments if max_len is None else max_len
    return (f"{cell},{shape},{distribution},{kind},sum,fp32,uint32,{d},{rows},{rows},{segments},"
            f"{rows / segments:.3f},{max_len},0,{input_bytes},{offset_bytes},{output_bytes},{moved},0.2,{passes},1,{us},"
            f"4,{samples},{us},{us * 0.9},{moved / us / 1e6:.4f},{0.9:.4f},{verified}")


def write(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


def args(**overrides):
    values = dict(target=1.5, spread_gate=1.05, copy_gate=0.85, large_mib=16.0, worst=5, copy=True)
    values.update(overrides)
    return argparse.Namespace(**values)


class AuditTest(unittest.TestCase):
    def setUp(self):
        mib32 = 8 * 2**20  # rows of 4-byte values: 32 MiB at D = 1
        self.path = write([
            '# ragged_reduction_audit=1 version=1263 dtype=fp32 device="Test GPU" sm_count=170 stamp=runtime '
            "copy_bandwidth_tbps=1.5900",
            "# copy_curve 2097152=2.0 4194304=3.5",
            HEADER,
            # shape 0: 32 MiB, uniform fast, one_giant 10x slower
            row(0, 0, "uniform", "value", 1, mib32, 1024, 25.0, "cub_segmented_reduce"),
            row(1, 0, "one_giant", "value", 1, mib32, 1024, 250.0, "cub_segmented_reduce", max_len=mib32, verified=0),
            # shape 1: 32 MiB at D = 64, distributions within 2%
            row(2, 1, "uniform", "arg_index", 64, mib32 // 64, 4096, 24.0, "vector_wide_arg_t1"),
            row(3, 1, "one_giant", "arg_index", 64, mib32 // 64, 4096, 24.5, "vector_wide_arg_t1"),
            # shape 2: small, below --large-mib
            row(4, 2, "uniform", "value", 4, 1024, 16, 3.0),
            "# failed,5,3,pareto,sum,D=1,rows=10,segments=2,unsupported",
            "# verify_failed,1,0,one_giant,sum,D=1,rows=8388608,segments=1024,mismatches=1/1024,segment 7: got 0",
            "# end cells=5 failed=1 verify_failed=1 copy_bandwidth_tbps_end=1.5800",
        ])

    def tearDown(self):
        os.unlink(self.path)

    def test_load(self):
        header, cells, failed, verify_failed = audit.load(self.path)
        self.assertEqual(header["dtype"], "fp32")
        self.assertEqual(header["device"], "Test GPU")
        self.assertEqual(header["verify_failed"], "1")
        self.assertEqual(len(cells), 5)
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(verify_failed), 1)
        self.assertAlmostEqual(cells[0]["spread"], 0.03, places=6)
        self.assertEqual(cells[1]["max_len"], 8 * 2**20)

    def test_spreads_compare_distributions_of_a_shape(self):
        _, cells, _, _ = audit.load(self.path)
        spreads = {s[0][0]["shape_id"]: s for s in audit.shape_spreads(cells, 16 * 2**20)}
        self.assertEqual(set(spreads), {0, 1})  # shape 2 is small and has one distribution
        self.assertAlmostEqual(spreads[0][3], 10.0, delta=0.01)
        self.assertEqual(spreads[0][2]["distribution"], "one_giant")
        self.assertAlmostEqual(spreads[1][3], 24.5 / 24.0, places=2)

    def test_report_names_every_section_and_the_gate(self):
        header, cells, failed, verify_failed = audit.load(self.path)
        out = io.StringIO()
        audit.report(header, cells, failed, verify_failed, args(), out)
        text = out.getvalue()
        self.assertIn("# gate (moved >= 16 MiB)", text)
        self.assertRegex(text, r"distribution spread p50 1\.02\d p90 9\.99\d max 9\.99\d over 2 shapes")
        self.assertIn("cells failing verification 1", text)
        self.assertIn("# gate of copy (moved >= 16 MiB): p50 0.900 p10 0.900 (gate 0.85); timed full", text)  # 1275
        for section in ("1. Moved TB/s by kind", "2. Moved TB/s by distribution", "3. Distribution spread",
                        "widest spreads", "4. Moved TB/s by D", "5. Moved TB/s by mean segment length",
                        "6. What production runs", "7. Slowest cells", "8. Stamp wall time", "9. Measurement",
                        "10. (--copy)"):
            self.assertIn(section, text)
        widest = text.split("widest spreads")[1].split("4. Moved")[0]
        self.assertLess(widest.index("one_giant"), widest.index("vector_wide_arg_t1"))
        self.assertIn("cub_segmented_reduce", text)
        self.assertIn("vector_wide_arg_t", text)  # signature with digits stripped
        self.assertIn("failed cells (1)", text)
        self.assertIn("cells failing verification (1)", text)

    def test_ragged_kr_candidates_keep_the_best_per_cell(self):
        mib32 = 8 * 2**20
        path = write([
            '# ragged_reduction_audit=1 version=1265 dtype=fp8_e4m3 device="Test GPU" backend=ragged_kr skipped=12',
            "# ragged_kr_candidate ragged_kr:i16_c0 grid=1020 workspace_bytes=65280",
            "# ragged_kr_candidate ragged_kr:i8_c0 grid=1360 workspace_bytes=87040",
            HEADER,
            row(0, 0, "uniform", "value", 1, mib32, 1024, 25.0, "ragged_kr:i16_c0+ragged_finish"),
            row(1, 0, "uniform", "value", 1, mib32, 1024, 30.0, "ragged_kr:i8_c0+ragged_finish"),
            row(2, 0, "one_giant", "value", 1, mib32, 1024, 26.0, "ragged_kr:i16_c0+ragged_finish"),
            row(3, 0, "one_giant", "value", 1, mib32, 1024, 24.0, "ragged_kr:i8_c0+ragged_finish", verified=0),
        ])
        try:
            header, rows, _, _ = audit.load(path)
            self.assertEqual(header["backend"], "ragged_kr")
            self.assertEqual(header["candidates"]["ragged_kr:i8_c0"]["grid"], "1360")
            best, by_key = audit.best_candidates(rows)
            self.assertEqual([c["cell_id"] for c in best], [0, 3])
            self.assertEqual(len(by_key[(0, "one_giant")]), 2)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(audit.main([path, "--compare", self.path]), 0)
            text = out.getvalue()
            self.assertIn("backend=ragged_kr skipped=12 cells=2", text)
            self.assertIn("cells failing verification 1", text)  # cell 3's candidate failed
            self.assertRegex(text, r"distribution spread p50 1\.04\d")  # 26 / 25 us at equal bytes
            candidates = text.split("12. Candidates")[1].split("11. This run")[0]
            self.assertIn("ragged_kr:i16_c0+ragged_finish", candidates)
            self.assertRegex(candidates, r"ragged_kr:i16_c0\+ragged_finish\s+2\s+1\s")  # 2 cells, 1 win
            self.assertIn("11. This run against the other (2 shared cells)", text)
        finally:
            os.unlink(path)

    def test_cost_fit_recovers_a_linear_model(self):  # 1272
        cells = []
        for i, (rows, segments) in enumerate([(2**24, 1000), (2**26, 70000), (2**25, 2**20), (2**27, 3), (2**24, 2**21)]):
            us = 12.0 + rows / 1.5e6 + segments * 15.0 / 1.5e6  # 1.5 TB/s rows; an end costs 15 rows
            for offsets in ("uint32", "uint64"):
                cells.append({"distribution": "uniform", "moved_bytes": rows + 9 * segments, "passes": "ragged_kr:x",
                              "offset_dtype": offsets, "D": 1, "rows": rows, "segments": segments, "median_us": us})
            # 1280: D = 3, fitted apart, rows counted in bytes (an end costs 40 bytes; no extra column).
            cells.append({"distribution": "uniform", "moved_bytes": 3 * rows + 9 * segments, "passes": "ragged_kr:x",
                          "offset_dtype": "uint32", "D": 3, "rows": rows, "segments": segments,
                          "median_us": 12.0 + 3 * rows / 1.5e6 + segments * 40.0 / 1.5e6})
        cells.append(dict(cells[0], distribution="one_giant", median_us=1e6))  # not a uniform cell: ignored
        out = io.StringIO()
        audit.cost_fit(cells, {"value_output": "fp32"}, out, min_bytes=16 * 2**20)
        text = out.getvalue()
        self.assertIn("13. Cost fit", text)
        self.assertRegex(text, r"ragged_kr:x\s+uint32\s+1\s+5\s+12\.00\s+1\.500\s+10\.0\s+15\.0\s+7\.0\s+0%")
        self.assertRegex(text, r"ragged_kr:x\s+uint64\s+1\s+5\s+12\.00\s+1\.500\s+10\.0\s+15\.0\s+3\.0\s+0%")
        self.assertRegex(text, r"ragged_kr:x\s+uint32\s+2-15\s+5\s+12\.00\s+1\.500\s+26\.7\s+40\.0\s+-\s+0%")
        # 1284: FP16 input: a row of D elements is 2 x D bytes, and the D >= 2 class is 2-7.
        wide = [dict(c, median_us=12.0 + 2 * c["D"] * c["rows"] / 1.5e6 + c["segments"] * 30.0 / 1.5e6)
                for c in cells if c["distribution"] == "uniform"]
        out = io.StringIO()
        audit.cost_fit(wide, {"value_output": "fp32", "dtype": "fp16"}, out, min_bytes=16 * 2**20)
        text = out.getvalue()
        self.assertRegex(text, r"ragged_kr:x\s+uint32\s+1\s+5\s+12\.00\s+1\.500\s+20\.0\s+30\.0\s+22\.0\s+0%")
        self.assertRegex(text, r"ragged_kr:x\s+uint32\s+2-7\s+5\s+12\.00\s+1\.500\s+20\.0\s+30\.0\s+-\s+0%")
        # 1285: FP8 rows of 16 bytes and more fit apart, as ">=16B".
        rk = [dict(c, D=64, passes="ragged_rk:x", median_us=12.0 + 64 * c["rows"] / 1.5e6 + c["segments"] * 300.0 / 1.5e6)
              for c in cells if c["distribution"] == "uniform" and c["D"] == 1]
        out = io.StringIO()
        audit.cost_fit(rk, {"value_output": "fp32", "dtype": "fp8_e4m3"}, out, min_bytes=16 * 2**20)
        self.assertRegex(out.getvalue(), r"ragged_rk:x\s+uint32\s+>=16B\s+5\s+12\.00\s+1\.500\s+200\.0\s+300\.0\s+-\s+0%")

    def test_compare_and_main(self):
        mib32 = 8 * 2**20
        other = write([HEADER, row(0, 0, "uniform", "value", 1, mib32, 1024, 50.0),
                       row(1, 0, "one_giant", "value", 1, mib32, 1024, 500.0),
                       row(2, 1, "uniform", "arg_index", 64, mib32 // 64, 9999, 24.0)])  # other segments: not shared
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(audit.main([self.path, "--compare", other]), 0)
            text = out.getvalue()
            self.assertIn("11. This run against the other (2 shared cells)", text)
            self.assertIn("by distribution", text)
            self.assertIn("cells at >= 1 MiB slower than the other run (0 of 2", text)
            self.assertIn("2.000", text)  # this run is twice as fast on both shared cells
            self.assertNotIn("10. (--copy)", text)
        finally:
            os.unlink(other)


if __name__ == "__main__":
    unittest.main()
