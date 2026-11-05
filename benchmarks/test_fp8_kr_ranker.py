"""Host-only checks for the KR ranker dataset (1158): parsing, skipped variants, and the heuristic list order."""
import gzip
import os
import pathlib
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fp8_ranker"))
sys.path.insert(0, os.path.join(HERE, "fp8_kr_ranker"))
import dataset  # noqa: E402  (fp8_kr_ranker)
import evaluate  # noqa: E402  (fp8_ranker)

COLUMNS = ("kind,cell_id,heldout,random_geometry,K,R,input_bytes,target_mib,dtype,operation,shape,family,variant,"
           "in_list,list_position,pool_position,abandoned,median_us," + ",".join(f"f{i}" for i in range(24)))


def row(shape, family, listed, pool, median, variant="full", abandoned=0):
    features = ",".join(str(float(i + pool)) for i in range(24))
    return (f"value,3,0,1,1000,8192,8192000,0,fp8_e4m3,sum,{shape},{family},{variant},{int(listed > 0)},{listed},"
            f"{pool},{abandoned},{median},{features}")


class TestKRRankerDataset(unittest.TestCase):
    def write(self, directory, lines, header="# fp8_kr_training=1 fp8_kr_ranker_training=1 feature_count=24"):
        path = pathlib.Path(directory) / "d.csv.gz"
        with gzip.open(path, "wt") as stream:
            stream.write(header + "\n" + COLUMNS + "\n" + "\n".join(lines) + "\n")
        return str(path)

    def test_load_and_list_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [row("b", "block", 1, 1, 10.0), row("g32", "group", 2, 2, 12.0),
                                          row("s4096", "split", 0, 3, 9.0), row("s4096", "split", 0, 0, 7.0,
                                                                                variant="partials"),
                                          row("g1", "group", 0, 4, 99.0, abandoned=1)])
            header, cells = dataset.load(path)
            self.assertEqual(header["feature_count"], "24")
            self.assertEqual(len(cells), 1)
            cell = cells[0]
            self.assertEqual([c.configuration for c in cell.candidates], ["b", "g32", "s4096", "g1"])
            self.assertEqual([c.list_position for c in cell.candidates], [1, 2, dataset.UNLISTED + 3,
                                                                          dataset.UNLISTED + 4])
            self.assertEqual(evaluate.list_order(cell), [0, 1, 2, 3])
            self.assertTrue(cell.candidates[3].abandoned)
            self.assertAlmostEqual(cell.oracle_ms, 9.0)  # the partials row is not a candidate
            self.assertEqual(cell.candidates[2].features[0], 3.0)
            self.assertEqual(evaluate.regret_curve(cell, evaluate.list_order(cell), 3)[-1], 1.0)

    def test_rejects_other_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [row("b", "block", 1, 1, 10.0)], header="# fp8_kr_training=1")
            with self.assertRaises(ValueError):
                dataset.load(path)

    def test_per_dtype_rankers(self):
        # 1182: one ranker per dtype; each trains on its dtype's cells and writes its own header and namespace.
        import train  # noqa: E402  (fp8_kr_ranker/train.py; standard library at import)
        self.assertEqual(sorted(train.DTYPES), ["fp32", "fp8"])
        fp8, fp32 = train.DTYPES["fp8"], train.DTYPES["fp32"]
        self.assertTrue(fp8["cells"]("fp8_e4m3") and fp8["cells"]("fp8_e5m2") and not fp8["cells"]("fp32"))
        self.assertTrue(fp32["cells"]("fp32") and not fp32["cells"]("fp8_e4m3"))
        self.assertEqual(fp32["model_header"], "Fp32KRRankerModel.h")
        self.assertIn("Fp32KR::Fp32KRRankerModel", fp32["model_namespace"])
        self.assertIn("Fp32KR::Fp32KRRankerParity", fp32["parity_namespace"])
        self.assertIn("FP32_KR_SHORTLIST_DEPTHS", fp32["depths"])
        self.assertEqual(fp8["model_header"], "Fp8KRRankerModel.h")
        self.assertIn("FP8_KR_SHORTLIST_DEPTHS", fp8["depths"])


if __name__ == "__main__":
    unittest.main()
