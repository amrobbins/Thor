"""Tests of dense_production_audit.py on a tiny synthetic thor_dense_production_audit output."""
import contextlib
import io
import os
import tempfile
import unittest

import dense_production_audit as audit

HEADER = ("cell_id,family,kind,operation,dtype,outer,R,K,input_bytes,moved_bytes,stamp_ms,passes,pass_count,probe_us,"
          "launches_per_sample,sample0_us,sample1_us,sample2_us,sample3_us,sample4_us,median_us,copy_us,tbps,of_copy")


def row(cell, family, kind, outer, r, k, us, copy_us, passes="direct:tiled_fixed_segment"):
    input_bytes = 4 * outer * r * k
    moved = input_bytes + 4 * outer * k
    samples = ",".join(str(us * f) for f in (0.99, 1.0, 1.0, 1.01, 1.02))
    return (f"{cell},{family},{kind},sum,fp32,{outer},{r},{k},{input_bytes},{moved},1.5,{passes},1,{us},4,{samples},"
            f"{us},{copy_us},{moved / us / 1e6:.4f},{copy_us / us:.4f}")


def write(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.path = write([
            '# dense_production_audit=1 version=1174 dtype=fp32 device="Test GPU" sm_count=170 '
            "copy_bandwidth_tbps=1.5900",
            "# copy_curve 2097152=2.0 4194304=3.5",
            HEADER,
            row(0, "R", "value", 1, 2**20, 1, 3.0, 2.8, "kparallel_p16"),            # 4 MiB, near copy speed
            row(1, "KR", "value", 4096, 8192, 1, 200.0, 85.0, "direct:contiguous_fixed_segment"),  # 128 MiB, slow
            row(2, "RK", "arg_index", 16, 4096, 1024, 300.0, 180.0, "fp8_arg_x12+direct:tiled_generic"),  # 256 MiB
            "# failed,3,RK,argmax,outer=1,R=2,K=2,unsupported",
            "# end cells=3 failed=1 copy_bandwidth_tbps_end=1.5800",
        ])

    def tearDown(self):
        os.unlink(self.path)

    def test_load(self):
        header, cells, failed = audit.load(self.path)
        self.assertEqual(header["dtype"], "fp32")
        self.assertEqual(header["device"], "Test GPU")
        self.assertEqual(header["copy_bandwidth_tbps_end"], "1.5800")
        self.assertEqual(len(cells), 3)
        self.assertEqual(len(failed), 1)
        self.assertAlmostEqual(cells[0]["spread"], 0.03, places=6)

    def test_signature_strips_digits(self):
        self.assertEqual(audit.signature("fp8_arg_x12+direct:tiled_generic"), "fp_arg_x+direct:tiled_generic")

    def test_report_names_every_section_and_the_slow_cell(self):
        header, cells, failed = audit.load(self.path)
        out = io.StringIO()
        audit.report(header, cells, failed, audit.main.__globals__["argparse"].Namespace(
            target=1.5, large_mib=16.0, worst=5, copy=True), out)
        text = out.getvalue()
        for section in ("1. Moved TB/s", "2. What production runs", "3. Slowest cells", "4. Stamp wall time",
                        "5. Measurement", "6. (--copy)"):
            self.assertIn(section, text)
        slow = text.split("3. Slowest cells")[1].split("4. Stamp")[0]
        self.assertLess(slow.index("direct:contiguous_fixed_segment"), slow.index("fp8_arg_x12"))
        self.assertIn("failed cells (1)", text)

    def test_compare_and_main(self):
        other = write([HEADER, row(0, "R", "value", 1, 2**20, 1, 6.0, 2.8), row(1, "KR", "value", 4096, 8192, 1,
                                                                                100.0, 85.0)])
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(audit.main([self.path, "--compare", other]), 0)
            text = out.getvalue()
            self.assertIn("7. This run against the other", text)
            self.assertNotIn("6. (--copy)", text)
            self.assertIn("2.000", text)  # R: this 3 us vs other 6 us, TB/s ratio 2
            self.assertIn("0.500", text)  # KR: this 200 us vs other 100 us
        finally:
            os.unlink(other)


if __name__ == "__main__":
    unittest.main()
