"""Tests of dense_copy_audit.py on a tiny synthetic thor_dense_copy_audit output."""
import contextlib
import io
import os
import tempfile
import unittest

import dense_copy_audit as audit

HEADER = ("cell_id,case,sub,dtype,dims,strides,plan,useful_bytes,touched_bytes,moved_bytes,verified,probe_us,launches,"
          "sample0_us,sample1_us,sample2_us,median_us,copy_us,ref_1387,ref_1387_us,tbps,of_copy,vs_1387")
# Through 1393 the audit also wrote gpuMatrixTranspose's columns (1394 deleted it); an older run still loads.
OLD_HEADER = ("cell_id,case,sub,dtype,dims,strides,plan,useful_bytes,touched_bytes,moved_bytes,verified,probe_us,"
              "launches,sample0_us,sample1_us,sample2_us,median_us,copy_us,ref_1387,ref_1387_us,matrix_transpose_us,"
              "matrix_transpose_verified,tbps,of_copy,vs_1387,vs_matrix_transpose")


def row(cell, case, moved, us, copy_us, ref_us, plan, sub="", verified=1, ref="gather", old=False):
    samples = ",".join(str(us * f) for f in (0.99, 1.0, 1.02))
    matrix_transpose = "30.0,1," if old else ""
    return (f"{cell},{case},{sub},fp32,64x32,1x64,{plan},{moved // 2},{moved // 2},{moved},{verified},{us},4,"
            f"{samples},{us},{copy_us},{ref},{ref_us},{matrix_transpose}{moved / us / 1e6:.4f},{copy_us / us:.4f},"
            f"{ref_us / us if ref_us else 0:.4f}" + (",1.2" if old else ""))


def write(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


class AuditTest(unittest.TestCase):
    def setUp(self):
        mib32 = 32 * 2**20
        self.path = write([
            '# dense_copy_audit=1 version=1389a dtype=fp32 device="Test GPU" sm_count=170 copy_bandwidth_tbps=1.5400',
            "# copy_curve 4096=2.0 8192=2.1",
            HEADER,
            row(0, "d", mib32, 22.0, 21.0, 22.0, "memcpy_w4", ref="memcpy"),
            row(1, "r", mib32, 25.0, 21.0, 200.0, "transpose_p1_w4_t32x32_b0"),
            row(2, "r", 2**16, 4.0, 2.0, 6.0, "transpose_p1_w4_t32x32_b0"),
            row(3, "k", mib32, 30.0, 21.0, 90.0, "row_copy_o2_p3_w4", sub="both"),
            row(4, "k", mib32, 28.0, 21.0, 60.0, "row_copy_o1_p5_w4", sub="row_gap", ref="memcpy_2d"),
            row(5, "s", mib32, 40.0, 21.0, 41.0, "row_copy_o1_p1_w4", sub="stride_33", verified=0),
            row(6, "g", mib32, 26.0, 21.0, 120.0, "transpose_p7_w4_t34x30_b0"),
            "# verify_failed,5,s,plan=row_copy_o1_p1_w4,mismatched_bytes=4",
            "# failed,7,n,dims=2x3,strides=3x1,out of memory",
            "# end cells=7 failed=1 unverified=1",
        ])
        self.slower = write(["# dense_copy_audit=1 version=1388 dtype=fp32", OLD_HEADER,
                             row(1, "r", mib32, 50.0, 21.0, 200.0, "transpose_p1_w4_t32_b0", old=True),
                             row(3, "k", mib32, 30.0, 21.0, 90.0, "row_copy_o2_p3_w4", sub="both", old=True)])

    def tearDown(self):
        os.unlink(self.path)
        os.unlink(self.slower)

    def run_report(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(audit.main([self.path, *args]), 0)
        return out.getvalue()

    def test_load(self):
        header, cells, failed, verify_failed = audit.load(self.path)
        self.assertEqual(header["dtype"], "fp32")
        self.assertEqual(header["unverified"], "1")
        self.assertEqual(len(cells), 7)
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(verify_failed), 1)
        self.assertAlmostEqual(cells[1]["vs_1387"], 8.0, places=3)
        self.assertAlmostEqual(cells[0]["spread"], 0.03, places=6)

    def test_plan_kind(self):
        self.assertEqual(audit.plan_kind("row_copy_o2_p3_w4"), "row_copy_o2_w4")
        self.assertEqual(audit.plan_kind("transpose_p17_w16_t32x8_b1"), "transpose_w16_b1")
        self.assertEqual(audit.plan_kind("transpose_p17_w16_t32_b1"), "transpose_w16_b1")  # a 1388 name
        self.assertEqual(audit.plan_kind("packed_gather_o5_w1"), "packed_gather_o5_w1")  # 1389b
        self.assertEqual(audit.plan_kind("matrix_transpose_w4"), "matrix_transpose_w4")  # 1389e
        self.assertEqual(audit.plan_kind("memcpy_w4"), "memcpy_w4")

    def test_report_sections(self):
        text = self.run_report()
        self.assertIn("# gate (moved >= 16 MiB)", text)
        self.assertIn("unverified cells: 1 [5]", text)
        self.assertIn("failed cells: 1", text)
        for section in ("# 1. by case", "# 2. by plan", "# 3. k by sub-kind", "# 5. the 15 cells", "# 6. measurement"):
            self.assertIn(section, text)
        self.assertNotIn("gpuMatrixTranspose", text)  # 1394
        self.assertIn("transpose_w4_b0", text)
        self.assertIn("k pitched packing      both", text)
        self.assertNotIn("# 7. compare", text)

    def test_load_older_run(self):  # 1394: a run with gpuMatrixTranspose's columns
        _, cells, _, _ = audit.load(self.slower)
        self.assertEqual(len(cells), 2)
        self.assertAlmostEqual(cells[0]["vs_1387"], 4.0, places=3)

    def test_compare(self):
        text = self.run_report("--compare", self.slower)
        self.assertIn("# 7. compare: the other run's time (revision 1388) over this run's (revision 1389a), 2 joined "
                      "cells", text)
        self.assertIn("plans that differ between the runs: 1 of 2", text)
        self.assertNotIn("is this the rebuilt library?", text)
        self.assertIn("r rotation             16-128M  cells    1 p50 2.000", text)
        same = self.run_report("--compare", self.path)
        self.assertIn("plans that differ between the runs: 0 of 7  (none: is this the rebuilt library?)", same)


if __name__ == "__main__":
    unittest.main()
