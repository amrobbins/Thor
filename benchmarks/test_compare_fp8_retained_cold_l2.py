"""Host-only deterministic correctness checks for the retained sweep comparator."""
import csv
import gzip
import pathlib
import tempfile
import unittest

from benchmarks.compare_fp8_retained_cold_l2 import read, paired, summarize_pairs, oracle_stats


class TestRetainedColdL2Compare(unittest.TestCase):
    def test_pairs_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            header = ['family','outer','R','K','dtype','configuration','operation',
                      'rank28','total_median_ms']
            paths=[]
            for name, times in [('before',[0.1,0.08]),('after',[0.09,0.07])]:
                path = root / f'{name}.csv.gz'
                with gzip.open(path,'wt') as stream:
                    stream.write('# output=fp32\n')
                    writer=csv.writer(stream)
                    writer.writerow(header)
                    for i, value in enumerate(times):
                        writer.writerow(['aligned_k',1000,512,64,'fp8_e4m3',f'p8_cta{32*(i+1)}_shards1',
                                         'sum_identity',i+1,value])
                paths.append(path)
            before,_,bmode=read(paths[0]); after,cells,amode=read(paths[1])
            self.assertEqual((bmode,amode),('fp32','fp32'))
            groups,missing_b,missing_a=paired(before,after)
            self.assertEqual((missing_b,missing_a),(0,0))
            self.assertEqual(len(groups['all']),2)
            self.assertIn('N=2',summarize_pairs(groups['all']))
            self.assertIn('exact28=1/1',oracle_stats(cells))


if __name__=='__main__':
    unittest.main()
