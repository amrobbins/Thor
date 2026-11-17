"""Tests of view_reduction_audit.py on a tiny synthetic thor_view_reduction_audit output."""
import contextlib
import io
import os
import tempfile
import unittest

import view_reduction_audit as audit

HEADER = ("cell_id,case,operation,dtype,dims,strides,axes,count,outputs,stride,pitch_bytes,span_bytes,useful_bytes,"
          "touched_bytes,stamp_ms,passes,reference,reference_passes,verified,max_rel_err,probe_us,launches,"
          "sample0_us,sample1_us,sample2_us,median_us,reference_us,copy_us,tbps,of_copy,of_reference")
# The H alternative columns
HEADER_1392 = HEADER + ",split_us,pitched_w1_us,pitched_w2_us,pitched_w4_us,pitched_w8_us,alternative_passes"


def row(cell, case, op, touched, us, ref_us, stride=0, verified=1, passes="physical:device_transform_reduce",
        reference="dense_stamp", dims="64x32", strides="1x64", axes="0x1"):
    samples = ",".join(str(us * f) for f in (0.99, 1.0, 1.02))
    of_ref = ref_us / us if ref_us else 0.0
    pitch = 4 * stride
    return (f"{cell},{case},{op},fp32,{dims},{strides},{axes},2048,1,{stride},{pitch},{touched},{touched},{touched},"
            f"0.5,"
            f"{passes},{reference},KR Pack,{verified},0,{us},4,{samples},{us},{ref_us},{us * 0.8},"
            f"{touched / us / 1e6:.4f},0.8,{of_ref:.4f}")


def alternatives(split="", w1="", w2="", w4="", w8=""):
    """The alternative columns (split, then W 1 / 2 / 4 / 8), appended to a row()."""
    return f",{split},{w1},{w2},{w4},{w8},x"


def write(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


class AuditTest(unittest.TestCase):
    def setUp(self):
        mib32 = 32 * 2**20
        self.path = write([
            '# view_reduction_audit=1 version=1335 dtype=fp32 device="Test GPU" sm_count=170 '
            "copy_bandwidth_tbps=1.5400",
            "# copy_curve 4096=2.0 8192=2.1",
            HEADER,
            row(0, "a", "sum", mib32, 30.0, 25.0),
            row(1, "a", "max", 2**16, 5.0, 4.0),
            row(2, "b", "sum", mib32, 60.0, 24.0, passes="physical:contiguous_fixed_segment"),
            row(3, "c", "sum", mib32, 40.0, 38.0, stride=64),
            row(4, "c", "mean", 4 + 4, 3.0, 0.0, stride=0),
            row(5, "c", "min", mib32, 20.0, 20.0, stride=2, verified=0),
            # D against the dense stamp, as an older run's tiled passes (cell 7 the dense stamp faster, cell 8 slower);
            # E unsupported by the view stamp (cell 9 times the dense stamp, no reference)
            row(7, "d", "sum", mib32, 30.0, 20.0, passes="tiled:permutation_natural"),
            row(8, "d", "max", 2**21, 4.0, 5.0, passes="tiled:permutation_natural"),
            row(9, "e", "sum", mib32, 25.0, 0.0, passes="retained_fp32_kr:s#+retained_fp32_rk:x",
                reference="view_unsupported"),
            "# verify_failed,5,c,min,count=2048,got=0,max_rel_err=1",
            "# failed,6,c,sum,dims=10,strides=3,out of memory",
            "# end cells=6 failed=1 unverified=1",
        ])
        faster = write([HEADER, row(0, "a", "sum", mib32, 20.0, 25.0), row(2, "b", "sum", mib32, 24.0, 24.0),
                        row(3, "c", "sum", mib32, 45.0, 38.0, stride=64)])
        self.faster = faster

    def tearDown(self):
        os.unlink(self.path)
        os.unlink(self.faster)

    def run_report(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            audit.main([self.path, *argv])
        return out.getvalue()

    def test_load(self):
        header, cells, failed, verify_failed = audit.load(self.path)
        self.assertEqual(header["dtype"], "fp32")
        self.assertEqual(header["device"], "Test GPU")
        self.assertEqual(len(cells), 9)
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(verify_failed), 1)
        self.assertAlmostEqual(cells[0]["of_reference"], 25.0 / 30.0, places=3)
        self.assertAlmostEqual(cells[0]["spread"], 0.03, places=6)

    def test_gate_and_sections(self):
        text = self.run_report()
        self.assertIn("# gate (touched >= 16 MiB)", text)
        self.assertIn("A full permutation", text)
        self.assertIn("unverified cells 1 (1 verify_failed lines), failed cells 1", text)
        self.assertIn("# 2. C by stride", text)
        self.assertIn("# 4. pass signatures", text)
        # the s = 0 cell has no reference and is left out of of_reference
        cells = audit.load(self.path)[1]
        self.assertEqual(len(audit.with_reference(cells)), 7)

    def test_compare_and_legacy_clause(self):
        text = self.run_report("--compare", self.faster)
        self.assertIn("# 8. compare", text)
        self.assertIn("(3 joined cells)", text)
        # cells 0 and 2: the other run took 20 / 24 us against this run's 30 / 60, past the 2% tolerance;
        # cell 3 is faster
        self.assertIn("legacy clause (touched >= 1 MiB, slower by more than 2%): 2 cells", text)

    def test_phase5(self):
        text = self.run_report()
        self.assertIn("# 9. D view / dense", text)
        # D: view / dense 1.5 (cell 7) and 0.8 (cell 8); p50 1.15, one cell each way
        self.assertIn("D all                        cells    2 view/dense p50 1.150", text)
        self.assertIn("dense faster   1 slower   1", text)
        # the dense stamp furthest behind the view first
        self.assertLess(text.index("dense slowest:     8"), text.index("dense slowest:     7"))
        self.assertIn("E cells 1, unsupported by the view stamp 1", text)
        self.assertIn("E dense 16-128M", text)

    def test_tiled_classes(self):
        # F / G / H against the dense reference (view / dense 30, 1.5 and 2.5); the two-part pass chains
        path = write([HEADER,
                      row(20, "f", "sum", 2**21, 300.0, 10.0,
                          passes="split_rotated:retained_fp32_kr:s4+copy:transpose_p1_w4_t32x32_b0"),
                      row(21, "f", "max", 2**21, 12.0, 10.0,
                          passes="split_rotated:retained_fp32_kr:s8+copy:transpose_p1_w4_t64x8_b0"),
                      row(22, "g", "sum", 2**21, 30.0, 20.0,
                          passes="split_rotated:retained_fp32_rk:x+copy:transpose_p3_w4_t32x32_b0"),
                      row(23, "h", "sum", 2**21, 50.0, 20.0,
                          passes="split_packed:copy:row_copy_o2_p5_w4+retained_fp32_kr:s2")])
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                audit.main([path, "--slowest", "2"])
            text = out.getvalue()
        finally:
            os.unlink(path)
        self.assertIn("# 10. F / G / H", text)
        self.assertIn("F all                                        cells    2 view/dense p50 15.615", text)
        self.assertIn(">=2x   1 >=10x   1", text)
        self.assertIn("F <=256", text)  # every synthetic row has one output
        # the family strips the tiles and sizes, so both F rows are one family
        self.assertIn("F rotated:retained_fp#_kr:s#+copy:transpose_ cells    2", text)
        self.assertIn("G rotated:retained_fp#_rk:x+copy:transpose_p cells    1", text)
        self.assertIn("H packed:copy:row_copy_o#_p#_w#+retained_fp# cells    1", text)
        self.assertIn("H R 1K-16K", text)  # count 2048
        # the slowest F cell first, with its family
        self.assertLess(text.index("slowest:    20"), text.index("slowest:    21"))
        self.assertIn("view/dense 30.0 rotated:retained_fp#_kr:s#+copy:transpose_", text)

    def test_h_mechanisms(self):
        # H by gaps and path. [1, 122, 30] at row pitch 31 (4-byte gaps), [4, 6, 8] at slab pitch 50, [3, 5, 2] at row
        # pitch 4 and slab pitch 25.
        big = 32 * 2**20
        path = write([HEADER,
                      row(30, "h", "sum", big, 12.0, 10.0, passes="split_sliced:retained_fp32_rk:x+copy:row_copy",
                          dims="122x30", strides="31x1", axes="0"),
                      row(31, "h", "sum", 2**20, 30.0, 10.0, passes="split_packed:copy:row_copy+retained_fp32_rk:x",
                          dims="4x6x8", strides="50x8x1", axes="1"),
                      row(32, "h", "max", 2**20, 25.0, 10.0, passes="split_packed:copy:row_copy+retained_fp32_rk:x",
                          dims="3x5x2", strides="25x4x1", axes="1")])
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                audit.main([path])
            text = out.getvalue()
        finally:
            os.unlink(path)
        self.assertIn("# 11. H by gaps and path", text)
        self.assertIn("row gaps < 32 B   sliced   all   cells   1 view/dense p50 1.200", text)
        self.assertIn("row gaps < 32 B   sliced   large cells   1", text)
        self.assertIn("slab gaps         packed   all   cells   1 view/dense p50 3.000", text)
        self.assertNotIn("slab gaps         packed   large", text)
        self.assertIn("row + slab gaps   packed   all   cells   1 view/dense p50 2.500", text)

    def test_pitched_alternatives(self):
        # 170 SMs. [2048, 4, 64] at pitches 400 / 80: 2048 items of 256-byte tiles, W 1 (four rows), 3.01 KiB per SM in
        # flight, read in place (10 us) where the split takes 30. [8, 6, 300] at 4000 / 320: 8 x 3 tiles of 400 bytes at
        # W 1 (0.055 KiB per SM), split (12 us) where W 8 would take 9.
        path = write(['# view_reduction_audit=1 version=1335 dtype=fp32 device="Test GPU" sm_count=170',
                      HEADER_1392,
                      row(40, "h", "sum", 2**20, 10.0, 9.0, passes="pitched_complete:w1_t64", dims="2048x4x64",
                          strides="400x80x1", axes="1") + alternatives(30.0, 10.5, 11.0, 12.0, 14.0),
                      row(41, "h", "sum", 2**20, 12.0, 9.0, passes="split_packed:copy:row_copy+retained_fp32_rk:x",
                          dims="8x6x300", strides="4000x320x1", axes="1") + alternatives("", 20.0, 14.0, 10.0, 9.0),
                      row(42, "a", "sum", 2**20, 12.0, 9.0) + alternatives()])
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                audit.main([path])
            text = out.getvalue()
        finally:
            os.unlink(path)
        self.assertIn("# 12. H in place / split by K tile bytes and bytes in flight per SM (170 SMs)", text)
        self.assertIn("tile >=256 B in flight/SM <2K  cells   1 in place   0 split/best W p50 1.333", text)
        self.assertIn("chosen/best p50 1.333", text)
        self.assertIn("best: split 0 w1 0 w2 0 w4 0 w8 1", text)
        self.assertIn("tile >=256 B in flight/SM 2-4K cells   1 in place   1 split/best W p50 3.000", text)
        self.assertIn("best: split 0 w1 1 w2 0 w4 0 w8 0", text)
        self.assertIn("choice slower than the best by more than 2%: 1 cells", text)
        self.assertIn("41 sum         O 8 R 6 I 300 W 1 in flight 0.055 KiB/SM ran split_packed 12.0 us, best"
                      " pitched_w8 9.0 us", text)

    def test_quantile(self):
        self.assertEqual(audit.quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertNotEqual(audit.quantile([], 0.5), audit.quantile([], 0.5))  # nan


if __name__ == "__main__":
    unittest.main()
