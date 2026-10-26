"""Host-only tests for the read-only production-object census classifier."""
import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmarks.fp8_selected_inventory_size import parse_func_symbols
from benchmarks.fp8_production_object_census import (
    classify_function, compare_selected_kernel_identities, demangle_func_symbols,
    family_count_discrepancies, load_selected_symbol_rows, selected_manifest_counts,
)


class ProductionObjectClassificationTest(unittest.TestCase):
    def test_retained_family_names(self):
        samples = {
            'fp8GeneralRParallelPackedCompleteKernel': 'OneWindow',
            'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'DeepRSmallK',
            'kParallelFp8PacketKernel': 'AlignedK',
            'fp8GeneralRParallelKTileP16CompleteKernel': 'RuntimeK',
        }
        for symbol, expected in samples.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(classify_function(
                    f'ThorImplementation::CubReductionInternal::{symbol}<__nv_fp8_e4m3>()'), expected)

    def test_retained_mangled_cuda_kernel_is_classified_without_demangling(self):
        for kernel, family in {
            'fp8GeneralRParallelPackedCompleteKernel': 'OneWindow',
            'narrowFp8PackedFlatRCooperativeFirstStageKernel': 'DeepRSmallK',
            'kParallelFp8PacketKernel': 'AlignedK',
            'awkwardAlignmentRotatedShardedFirstStageKernel': 'RuntimeK',
        }.items():
            mangled = (f'_ZN18ThorImplementation20CubReductionInternal'
                       f'{len(kernel)}{kernel}I13__nv_fp8_e4m3E')
            self.assertEqual(classify_function(mangled, mangled), family)

    def test_shared_rotated_non_fp8_kernels_are_not_mislabeled_runtime_k(self):
        name = ('void ThorImplementation::CubReductionInternal::'
                '(anonymous namespace)::awkwardAlignmentRotatedShardedFirstStageKernel'
                '<float, PropagatingMaximumFp32>()')
        self.assertEqual(classify_function(name), 'other-device-function')
        typed = name.replace('<float,', '<__nv_fp8_e4m3,')
        self.assertEqual(classify_function(typed), 'RuntimeK')

    def test_other_mangled_fp8_kernel_remains_visible(self):
        mangled = '_ZN18ThorImplementation18legacyFp8KernelI...'
        self.assertEqual(classify_function(mangled, mangled), 'other-fp8-kernel')

    def test_unselected_fp8_kernel_is_visible(self):
        self.assertEqual(classify_function(
            'ThorImplementation::CubReductionInternal::legacyFp8Kernel<__nv_fp8_e5m2>()'),
            'other-fp8-kernel')
        self.assertEqual(classify_function(
            'ThorImplementation::CubReductionInternal::fp8UnretainedKernel()'),
            'other-fp8-kernel')

    def test_helper_and_non_fp8_kernel_are_not_false_positives(self):
        self.assertEqual(classify_function(
            'ThorImplementation::CubReductionInternal::loadFp8Packet<__nv_fp8_e4m3>()'),
            'other-device-function')
        self.assertEqual(classify_function(
            'ThorImplementation::CubReductionInternal::reduceKernel<float>()'),
            'other-device-function')
        self.assertEqual(classify_function(
            'cub::detail::device::somethingKernel<__nv_fp8_e4m3>()'),
            'other-device-function')


class ProductionObjectManifestComparisonTest(unittest.TestCase):
    def test_reports_family_name_surplus_without_claiming_unreachable_kernels(self):
        selected = Counter({'OneWindow': 1008, 'DeepRSmallK': 840,
                            'AlignedK': 160, 'RuntimeK': 312})
        production = Counter({'OneWindow': 1876, 'DeepRSmallK': 840,
                              'AlignedK': 436, 'RuntimeK': 528})
        self.assertEqual(family_count_discrepancies(production, selected), {
            'OneWindow': 868, 'DeepRSmallK': 0, 'AlignedK': 276, 'RuntimeK': 216,
        })

    def test_reads_generated_selector_manifest_not_hardcoded_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            build = Path(folder)
            self.assertIsNone(selected_manifest_counts(build))
            generated = build / 'fp8-selected-inventory'
            generated.mkdir()
            (generated / 'expected.json').write_text(json.dumps([
                {'mode': 'OneWindow'}, {'mode': 'OneWindow'}, {'mode': 'RuntimeK'}]))
            self.assertEqual(selected_manifest_counts(build),
                             Counter({'OneWindow': 2, 'RuntimeK': 1}))

    def test_rejects_unrecognized_manifest_family(self):
        with tempfile.TemporaryDirectory() as folder:
            generated = Path(folder) / 'fp8-selected-inventory'
            generated.mkdir()
            (generated / 'expected.json').write_text('[{"mode": "NotASelectedFamily"}]')
            with self.assertRaisesRegex(RuntimeError, 'Invalid selected-inventory manifest'):
                selected_manifest_counts(Path(folder))


class ProductionObjectIdentityComparisonTest(unittest.TestCase):
    @staticmethod
    def _prod(family, symbol, *, arch='sm_120',
              obj='CubReductionNarrowFlatRCooperative.cu.o', name=None):
        return {'category': family, 'mangled_name': symbol,
                'name': name or f'call_{symbol}()',
                'object': obj, 'cubin': f'cuda.{arch}.cubin'}

    @staticmethod
    def _selected(family, symbol, *, arch='sm_120', name=None):
        return {'mode': family, 'mangled_name': symbol,
                'name': name or f'call_{symbol}()',
                'object': 'narrow', 'cubin': f'narrow.{arch}.cubin'}

    def test_separates_new_kernel_duplicates_missing_and_architectures(self):
        production = [
            self._prod('OneWindow', '_Zselected', obj='first/CubReductionNarrowFlatRCooperative.cu.o'),
            self._prod('OneWindow', '_Zselected', obj='second/CubReductionNarrowFlatRCooperative.cu.o'),
            self._prod('OneWindow', '_Zextra'),
            self._prod('RuntimeK', '_Zruntime', arch='sm_89'),
            self._prod('other-device-function', '_Zignored'),
        ]
        selected = [
            self._selected('OneWindow', '_Zselected'),
            self._selected('DeepRSmallK', '_Zmissing'),
            self._selected('RuntimeK', '_Zruntime', arch='sm_120'),
        ]
        comparisons, states = compare_selected_kernel_identities(production, selected)
        by_key = {(r['architecture'], r['family'], r['mangled_name']): r
                  for r in comparisons}
        self.assertEqual(by_key[('sm_120', 'OneWindow', '_Zselected')]['status'],
                         'repeated-in-production')
        self.assertEqual(by_key[('sm_120', 'OneWindow', '_Zselected')]['production_instances'], 2)
        self.assertEqual(by_key[('sm_120', 'OneWindow', '_Zextra')]['status'], 'production-only')
        self.assertEqual(by_key[('sm_120', 'DeepRSmallK', '_Zmissing')]['status'], 'selected-missing')
        self.assertEqual(by_key[('sm_89', 'RuntimeK', '_Zruntime')]['status'], 'production-only')
        self.assertEqual(by_key[('sm_120', 'RuntimeK', '_Zruntime')]['status'], 'selected-missing')
        self.assertEqual(states[('OneWindow', 'matched_instances')], 1)
        self.assertEqual(states[('OneWindow', 'repeated_instances')], 1)
        self.assertEqual(states[('OneWindow', 'production_only_instances')], 1)
        self.assertEqual(states[('RuntimeK', 'missing_selected_instances')], 1)
        self.assertEqual(len(comparisons), 5)

    def test_distinct_anonymous_namespace_salts_match_same_full_typed_signature(self):
        # A generated isolated CUDA TU has a different _GLOBAL__N__ hash from
        # the production TU.  The complete demangled signature is identical.
        signature = ('void ThorImplementation::CubReductionInternal::'
                     '(anonymous namespace)::fp8GeneralRParallelPackedCompleteKernel'
                     '<__nv_fp8_e4m3, 16, 16, 1, -1, 0>()')
        production = [self._prod('OneWindow', '_Zproduction_salt', name=signature)]
        isolated = [self._selected('OneWindow', '_Zisolated_salt', name=signature)]
        rows, states = compare_selected_kernel_identities(production, isolated)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'matched')
        self.assertEqual(rows[0]['production_mangled_names'], '_Zproduction_salt')
        self.assertEqual(rows[0]['selected_mangled_names'], '_Zisolated_salt')
        self.assertEqual(states['OneWindow', 'matched_instances'], 1)

    def test_different_template_argument_is_distinct_specialization(self):
        root = 'void ThorImplementation::CubReductionInternal::(anonymous namespace)::Kernel'
        production = [self._prod('AlignedK', '_Za',
                                 obj='CubReductionKParallelShardedTiled.cu.o',
                                 name=root + '<__nv_fp8_e4m3, 8>()')]
        selected = [self._selected('AlignedK', '_Zb', name=root + '<__nv_fp8_e4m3, 16>()')]
        # An isolated AlignedK symbol must originate from its aligned TU.
        selected[0]['object'] = 'aligned'
        rows, states = compare_selected_kernel_identities(production, selected)
        self.assertEqual({r['status'] for r in rows}, {'production-only', 'selected-missing'})
        self.assertEqual(states['AlignedK', 'matched_instances'], 0)

    def test_same_signature_from_unrelated_object_does_not_match(self):
        signature = 'void ThorImplementation::CubReductionInternal::Kernel<int>()'
        rows, _ = compare_selected_kernel_identities([
            self._prod('OneWindow', '_Zprod', obj='Unrelated.cu.o', name=signature)
        ], [self._selected('OneWindow', '_Zsel', name=signature)])
        self.assertEqual({r['status'] for r in rows}, {'production-only', 'selected-missing'})

    def test_missing_exact_names_fails_instead_of_guessing(self):
        production = [self._prod('AlignedK', '_Zkernel')]
        selected = [self._selected('AlignedK', '')]
        with self.assertRaisesRegex(RuntimeError, 'raw mangled names'):
            compare_selected_kernel_identities(production, selected)

    def test_requires_fresh_selected_symbol_report(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp)
            self.assertIsNone(load_selected_symbol_rows(build))
            report = build / 'fp8-selected-inventory/report/device-functions.csv'
            report.parent.mkdir(parents=True)
            report.write_text('object,cubin,mode,name\nnarrow,narrow.sm_120.cubin,OneWindow,K\n')
            with self.assertRaisesRegex(RuntimeError, 'outdated'):
                load_selected_symbol_rows(build)
            with report.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    'object', 'cubin', 'mode', 'name', 'mangled_name'])
                writer.writeheader()
                writer.writerow(dict(object='narrow', cubin='narrow.sm_120.cubin',
                                     mode='OneWindow', name='K', mangled_name='_Zkernel'))
            self.assertEqual(load_selected_symbol_rows(build)[0]['mangled_name'], '_Zkernel')


class ReadelfSymbolParserTest(unittest.TestCase):
    def test_extended_section_index_prefix_is_not_part_of_kernel_identity(self):
        text = (" 9583: 0000000000001234 256 FUNC LOCAL DEFAULT [ 10] 9583 "
                "_ZN18ThorImplementation6KernelEv\n")
        self.assertEqual(parse_func_symbols(text),
                         [('_ZN18ThorImplementation6KernelEv', 256)])

    def test_ordinary_readelf_elf_name_is_preserved(self):
        text = ' 12: 0000000000000000 128 FUNC LOCAL DEFAULT 10 _Z3foov\n'
        self.assertEqual(parse_func_symbols(text), [('_Z3foov', 128)])


class ProductionObjectDemanglingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++filt'), 'c++filt is not installed')
    def test_real_demangling_preserves_order(self):
        self.assertEqual(
            demangle_func_symbols([('_Z3foov', 8), ('_Z3barv', 16)], 'test ELF'),
            ['foo()', 'bar()'],
        )

    def test_bounded_arguments_preserve_long_symbol_order(self):
        mangled = [f'_Z{i}kernelv' for i in range(300)]
        symbols = [(name, 64) for name in mangled]
        calls = []

        def fake_run(cmd, **kwargs):
            self.assertEqual(cmd[:3], ['c++filt', '-n', '--'])
            self.assertNotIn('input_text', kwargs)
            self.assertLessEqual(len(cmd) - 3, 128)
            calls.append(cmd)
            return ''.join(f'demangled_{name}()\n' for name in cmd[3:])

        with patch('benchmarks.fp8_production_object_census.run', side_effect=fake_run):
            names = demangle_func_symbols(symbols, 'test object')
        self.assertEqual(names, [f'demangled_{name}()' for name in mangled])
        self.assertGreater(len(calls), 2)

    def test_unframed_batch_retries_each_symbol_without_dropping_rows(self):
        symbols = [('_Z3foov', 16), ('_Z3barv', 32), ('_Z3bazv', 48)]
        seen = []

        def fake_run(cmd, **kwargs):
            names = cmd[3:]
            seen.append(names)
            if len(names) > 1:
                return 'foo()\nbar()\n'  # One output is missing: invalid batch framing.
            return {'_Z3foov': 'foo()\n', '_Z3barv': 'bar()\n',
                    '_Z3bazv': 'baz()\n'}[names[0]]

        with patch('benchmarks.fp8_production_object_census.run', side_effect=fake_run):
            self.assertEqual(demangle_func_symbols(symbols, 'test cubin'),
                             ['foo()', 'bar()', 'baz()'])
        self.assertEqual(seen[1:], [['_Z3foov'], ['_Z3barv'], ['_Z3bazv']])

    def test_irrecoverable_missing_demangled_symbol_fails_loudly(self):
        with patch('benchmarks.fp8_production_object_census.run', return_value='\n'):
            with self.assertRaisesRegex(RuntimeError, 'test cubin'):
                demangle_func_symbols([('_Z3foov', 8), ('_Z3barv', 12)], 'test cubin')


if __name__ == '__main__':
    unittest.main()
