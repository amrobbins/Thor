"""Host checks for the ARG-A0 sweep additions and arg_value_compare.py."""
import contextlib
import io
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arg_value_compare  # noqa: E402

SOURCE = (HERE / 'Fp8RetainedColdL2Sweep.cpp').read_text()
HEADER = ('family,outer,R,K,dtype,configuration,operation,rank12,rank16,rank20,rank24,rank28,rank32,rank40,rank48,'
          'rank128,total_median_ms,sample0_ms,sample1_ms,sample2_ms,sample3_ms,sample4_ms\n')


def row(family, outer, r, k, config, op, rank, ms):
    ranks = ','.join(str(rank if rank and rank <= b else 0) for b in (12, 16, 20, 24, 28, 32, 40, 48, 128))
    return f'{family},{outer},{r},{k},fp8_e4m3,{config},{op},{ranks},{ms},{ms},{ms},{ms},{ms},{ms}\n'


def write_csv(directory, name, lines, arg_outputs='value_index'):
    path = pathlib.Path(directory) / name
    path.write_text(f'# cross_family_oracle=1 output=fp32 shapes=all arg_index=uint32 arg_outputs={arg_outputs} '
                    f'arg_value_output=fp32 dtypes=fp8_e4m3\n' + HEADER + ''.join(lines))
    return str(path)


class SweepSourceTest(unittest.TestCase):
    def test_arg_options_present(self):
        for text in ('"argmin"', '"argmax"', '--focus=arg', '--shapes=all', '--arg-outputs=index',
                     'analyzeDenseGeometry(dims,{1})', 'runOnInputForBenchmark', 'getPlannedPassNames',
                     '"arg_production"', '"one_window_arg"', 'launchFp8OneWindowArgComplete',
                     '"deep_r_small_k_arg"', 'launchFp8DeepRSmallKArgFirstStage', '"runtime_k_arg"',
                     'launchFp8RuntimeKArgPass', '"aligned_k_arg"', 'launchFp8AlignedKArgPass'):
            self.assertIn(text, SOURCE)

    def test_arg_ops_only_when_named(self):
        # The default sweep (no --ops) must not gain ARG cells: the ARG loop requires the name in settings.operations.
        arg_loop = SOURCE.split('for (const ArgOp& op : arg_ops) {', 1)[1]
        self.assertTrue(arg_loop.lstrip().startswith(
            'if (std::find(settings.operations.begin(),settings.operations.end(),op.name)==settings.operations.end())'))


class CompareTest(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            arg_value_compare.main(list(argv))
        return out.getvalue()

    def test_pairs_with_value_oracle_and_top28(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [
                row('one_window', 1000, 64, 32, 'o1_w1_p16', 'min_identity', 1, 0.050),
                row('aligned_k', 1000, 64, 32, 'p8', 'min_identity', 0, 0.040),  # oracle outside top28
                row('arg_production', 1000, 64, 32, 'tiled_narrow_async', 'argmin', 0, 0.100),
                row('arg_production', 1000, 64, 64, 'tiled_generic_rowlanes1', 'argmax', 0, 0.300),
                row('runtime_k', 1000, 64, 64, 'packet_complete', 'max_identity', 3, 0.100),
            ]
            text = self.run_main(write_csv(d, 'a.csv', lines))
            self.assertIn('paired_cells=2', text)
            # argmin 0.1/0.04 = 2.5, argmax 0.3/0.1 = 3.0 -> geomean sqrt(7.5)
            self.assertIn('geomean 2.739', text)
            self.assertIn('tiled_generic_rowlanes1', text)
            records, _ = arg_value_compare.paired_cells(*arg_value_compare.read(write_csv(d, 'b.csv', lines)))
            argmin = next(r for r in records if r['operation'] == 'argmin')
            self.assertAlmostEqual(argmin['oracle_ms'], 0.040)
            self.assertAlmostEqual(argmin['top28_ms'], 0.050)
            # Traffic: 1000*64*32 input bytes + 1000*32*(4 value + 4 index) output bytes in 0.1 ms.
            self.assertAlmostEqual(argmin['arg_tbps'], (1000 * 64 * 32 + 1000 * 32 * 8) / (0.1 * 1e9))

    def test_replicate_cells_are_averaged(self):
        # 24 and 96 MiB targets can both round to outer=1: the same cell measured twice.
        with tempfile.TemporaryDirectory() as d:
            lines = [row('arg_production', 1, 22528, 3071, 'tiled_generic_rowlanes1', 'argmax', 0, 0.2),
                     row('runtime_k', 1, 22528, 3071, 'packet_complete', 'max_identity', 1, 0.1),
                     row('arg_production', 1, 22528, 3071, 'tiled_generic_rowlanes1', 'argmax', 0, 0.4),
                     row('runtime_k', 1, 22528, 3071, 'packet_complete', 'max_identity', 1, 0.1)]
            records, _ = arg_value_compare.paired_cells(*arg_value_compare.read(write_csv(d, 'a.csv', lines)))
            self.assertEqual(1, len(records))
            self.assertAlmostEqual(0.3, records[0]['arg_ms'])
            self.assertAlmostEqual(0.1, records[0]['oracle_ms'])

    def test_twin_report(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [row('arg_production', 1000, 8, 2, 'fp8_arg_one_window:o8_w1_p16_c1', 'argmin', 0, 0.11),
                     row('one_window_arg', 1000, 8, 2, 'o8_w1_p16_c1', 'argmin', 0, 0.105),
                     row('one_window_arg', 1000, 8, 2, 'o4_w1_p16_c1', 'argmin', 0, 0.2),
                     row('one_window', 1000, 8, 2, 'o8_w1_p16', 'min_identity', 1, 0.1),
                     row('one_window_rk', 1000, 8, 2, 'o8_w1_p16_c1', 'min_identity', 0, 0.1),
                     row('one_window_rk', 1000, 8, 2, 'o4_w1_p16_c1', 'min_identity', 0, 0.1)]
            text = self.run_main(write_csv(d, 'a.csv', lines))
            self.assertIn('one_window_arg vs one_window_rk: 1 cells; best ARG twin / best VALUE candidate geomean 1.050', text)
            self.assertIn('fp8_arg_one_window', text)
            records, _ = arg_value_compare.paired_cells(*arg_value_compare.read(write_csv(d, 'b.csv', lines)))
            self.assertAlmostEqual(0.11, records[0]['arg_ms'])

    def test_deep_r_twin_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [row('arg_production', 10, 8192, 21, 'fp8_arg_deep_r_small_k:p16_rows512+direct:tiled_narrow',
                         'argmax', 0, 0.08),
                     row('deep_r_small_k_arg', 10, 8192, 21, 'p16_rows512', 'argmax', 0, 0.08),
                     row('deep_r_small_k', 10, 8192, 21, 'p16_rows512', 'max_identity', 1, 0.064)]
            text = self.run_main(write_csv(d, 'a.csv', lines))
            self.assertIn('deep_r_small_k_arg vs deep_r_small_k: 1 cells; best ARG twin / best VALUE candidate '
                          'geomean 1.250', text)
            self.assertIn('fp8_arg_deep_r_small_k+direct:tiled_narrow', text)

    def test_runtime_k_best_only(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [row('arg_production', 4, 24576, 2049, 'fp8_arg_runtime_k:p8_rows1024+direct:tiled_generic',
                         'argmin', 0, 0.2),
                     row('runtime_k_arg', 4, 24576, 2049, 'p8_rows1024', 'argmin', 0, 0.2),
                     row('runtime_k', 4, 24576, 2049, 'rotated_prefix_w0_tile497_rows1024', 'min_identity', 1, 0.125)]
            text = self.run_main(write_csv(d, 'a.csv', lines))
            self.assertIn('runtime_k_arg vs runtime_k: 1 cells; best ARG twin / best VALUE candidate geomean 1.600', text)

    def test_aligned_k_twin_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [row('arg_production', 64, 512, 1024, 'fp8_arg_aligned_k:p8_cta64_shards4+direct:tiled_generic',
                         'argmin', 0, 0.11),
                     row('aligned_k_arg', 64, 512, 1024, 'p8_cta64_shards4', 'argmin', 0, 0.11),
                     row('aligned_k', 64, 512, 1024, 'p8_cta64_shards4', 'min_identity', 1, 0.1)]
            text = self.run_main(write_csv(d, 'a.csv', lines))
            self.assertIn('aligned_k_arg vs aligned_k: 1 cells; best ARG twin / best VALUE candidate geomean 1.100', text)
            self.assertIn('p8_cta64_shards4', text)

    def test_index_only_traffic(self):
        options = {'output': 'fp32', 'arg_outputs': 'index'}
        self.assertEqual(4, arg_value_compare.output_bytes_per_element(options, True))
        self.assertEqual(4, arg_value_compare.output_bytes_per_element(options, False))
        self.assertEqual(5, arg_value_compare.output_bytes_per_element({'output': 'same_fp8'}, True))

    def test_after_comparison_and_mismatch_guard(self):
        with tempfile.TemporaryDirectory() as d:
            base = [row('arg_production', 100, 8, 2, 'tiled_narrow_async', 'argmin', 0, 0.2),
                    row('one_window', 100, 8, 2, 'o8_w1_p16', 'min_identity', 1, 0.1)]
            after = [row('arg_production', 100, 8, 2, 'fp8_arg_one_window', 'argmin', 0, 0.1),
                     row('one_window', 100, 8, 2, 'o8_w1_p16', 'min_identity', 1, 0.1)]
            text = self.run_main(write_csv(d, 'a.csv', base), '--after', write_csv(d, 'b.csv', after))
            self.assertIn('ARG after/before: 1 cells, geomean 0.500', text)
            with self.assertRaises(SystemExit):
                self.run_main(write_csv(d, 'c.csv', base), '--after', write_csv(d, 'e.csv', after, 'index'))


if __name__ == '__main__':
    unittest.main()
