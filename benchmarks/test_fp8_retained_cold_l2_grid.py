"""Stable quick gate for the permanent retained FP8 RK performance census."""
import pathlib
import re
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = (HERE / 'Fp8RetainedColdL2Sweep.cpp').read_text()


def census_shapes():
    # Mirrors the constant R/K arrays and seed geometry pairs in the source.
    explicit = SOURCE.split('std::vector<Shape> result{', 1)[1].split('};', 1)[0]
    seeds = {(int(r), int(k)) for r, k in re.findall(r'\{(\d+),(\d+)\}', explicit)}
    arrays = SOURCE.split('constexpr std::array<uint64_t, 36> rs{{', 1)[1]
    rs_text, remaining = arrays.split('}};', 1)
    ks_text = remaining.split('constexpr std::array<uint64_t, 28> ks{{', 1)[1].split('}};', 1)[0]
    rs = list(map(int, re.findall(r'\d+', rs_text)))
    ks = list(map(int, re.findall(r'\d+', ks_text)))
    for r in rs:
        for k in ks:
            x = ((r * 11400714819323198485) ^ (k * 14029467366897019727)) & ((1 << 64) - 1)
            if x % 7 == 0:
                seeds.add((r, k))
    return seeds


class RetainedFp8GridTest(unittest.TestCase):
    def test_historical_shape_and_cell_count(self):
        self.assertEqual(247, len(census_shapes()))
        self.assertEqual(17784, len(census_shapes()) * 3 * 2 * 12)

    def test_layout_and_batch_guards(self):
        self.assertIn('Fp8BenchmarkArenaCursor', SOURCE)
        self.assertIn('Fp8OracleBatchQueue queue(128,Passes', SOURCE)
        self.assertIn('constexpr int Passes = 5;', SOURCE)
        self.assertIn('std::sort(sorted.begin(),sorted.end())', SOURCE)
        self.assertIn('T::DataType::FP32', SOURCE)
        self.assertIn('makeSelectedIsolatedBenchmark(', SOURCE)
        self.assertIn('generalShortlist(', SOURCE)


if __name__ == '__main__':
    unittest.main()
