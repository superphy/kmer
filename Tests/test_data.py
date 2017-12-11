import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil
from data import get_kmer, get_genome_regions, get_omnilog_data
import numpy as np
import pandas as pd
import tempfile
import constants

class GetKmer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = self.dir + '/TEMPdatabase'
        self.metadata = self.dir + '/TEMPmetadata'
        file_header = 'Fasta,Class,Dataset\n'
        file_contents = 'A1,A,Train\nA2,A,Test\nB1,B,Train'
        with open(self.metadata, 'w') as f:
            f.write('%s%s' %(file_header, file_contents))
        with open(self.dir + '/A1', 'w') as f:
            f.write('>\nAAACCCCAA')
        with open(self.dir + '/A2', 'w') as f:
            f.write('>\nAACCCCAA')
        with open(self.dir + '/B1', 'w') as f:
            f.write('>\nAACCAACC')
        kwargs = {'metadata': self.metadata, 'prefix': self.dir + '/'}
        self.data = get_kmer(kwargs, self.db, recount=True, k=2, l=1)
        self.correct_x_train = np.array([[3,1,1,3],[2,2,1,2]])
        self.correct_y_train = np.array(['A', 'B'])
        self.correct_x_test = np.array([[2,1,1,3]])
        self.correct_y_test = np.array(['A'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_x_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_x_train:
            count2 += 1
            if elem in self.data[0]:
                count1 += 1
        for elem in self.data[0]:
            count2 += 1
            if elem in self.correct_x_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_y_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_y_train:
            count2 += 1
            if elem in self.data[1]:
                count1 += 1
        for elem in self.data[1]:
            count2 += 1
            if elem in self.correct_y_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_x_test(self):
        val = False
        if np.array_equal(self.data[2], self.correct_x_test):
            val = True
        self.assertTrue(val)

    def test_y_test(self):
        val = False
        if np.array_equal(self.data[3], self.correct_y_test):
            val = True
        self.assertTrue(val)


class GetGenomeRegion(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.table = self.dir + '/TEMPbinarytable'
        self.metadata = self.dir + '/TEMPmetadata'
        file_header = 'Fasta,Class,Dataset\n'
        file_contents = 'A1,A,Train\nA2,A,Test\nB1,B,Train'
        table_header = ',A1,A2,B1\n'
        table_contents = 'a,1,1,0\nb,1,0,1\nc,0,0,1\nd,1,1,1\ne,0,1,0\nf,1,0,1'
        with open(self.metadata, 'w') as f:
            f.write('%s%s' %(file_header, file_contents))
        with open(self.table, 'w') as f:
            f.write('%s%s' %(table_header, table_contents))
        kwargs = {'metadata': self.metadata}
        self.data = get_genome_regions(kwargs, table=self.table, sep=None)
        self.correct_x_train = np.array([[1,1,0,1,0,1],[0,1,1,1,0,1]])
        self.correct_y_train = np.array(['A', 'B'])
        self.correct_x_test = np.array([[1,0,0,1,1,0]])
        self.correct_y_test = np.array(['A'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_x_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_x_train:
            count2 += 1
            if elem in self.data[0]:
                count1 += 1
        for elem in self.data[0]:
            count2 += 1
            if elem in self.correct_x_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_y_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_y_train:
            count2 += 1
            if elem in self.data[1]:
                count1 += 1
        for elem in self.data[1]:
            count2 += 1
            if elem in self.correct_y_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_x_test(self):
        val = False
        if np.array_equal(self.data[2], self.correct_x_test):
            val = True
        self.assertTrue(val)

    def test_y_test(self):
        val = False
        if np.array_equal(self.data[3], self.correct_y_test):
            val = True
        self.assertTrue(val)

class GetOmnilog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.omnilog = self.dir + 'TEMPomnilog'
        self.metadata = self.dir + 'TEMPmetadata'
        with open(self.omnilog, 'w') as f:
            f.write(',A1,B1,A2,B2,B3,B4,A3,B5,A4,A5\n')
            f.write('a,3.1,3.3,3.4,3.53,2.7,2.8,2.43,4.1,4.2,3.3\n')
            f.write('b,1,0,1,1,1,1,1,1,0,0\n')
            f.write('c,0.01,0.2,0.3,0.34,0.50,0.02,0.1,0.34,0.56,0.23\n')
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            f.write('A1,A,Train\nA2,A,Train\nA3,A,Train\nA4,A,Train\nA5,A,Test\n')
            f.write('B1,B,Train\nB2,B,Train\nB3,B,Train\nB4,B,Test\nB5,B,Test\n')
        kwargs = {'metadata':self.metadata}
        self.data = get_omnilog_data(kwargs, omnilog_sheet=self.omnilog)
        self.correct_x_train = np.array([[3.1,1,0.01],[3.4,1,0.3],[2.43,1,0.1],
                                         [4.2,0,0.56],[3.3,0,0.2],[3.53,1,0.34],
                                         [2.7,1,0.50]])
        self.correct_y_train = np.array(['A','A','A','A','B','B','B'])
        self.correct_x_test = np.array([[3.3,0,0.23],[2.8,1,0.02],[4.1,1,0.34]])
        self.correct_y_test = np.array(['A','B','B'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_x_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_x_train:
            count2 += 1
            if elem in self.data[0]:
                count1 += 1
        for elem in self.data[0]:
            count2 += 1
            if elem in self.correct_x_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_y_train(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_y_train:
            count2 += 1
            if elem in self.data[1]:
                count1 += 1
        for elem in self.data[1]:
            count2 += 1
            if elem in self.correct_y_train:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_x_test(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_x_test:
            count2 += 1
            if elem in self.data[2]:
                count1 += 1
        for elem in self.data[2]:
            count2 += 1
            if elem in self.correct_x_test:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_y_test(self):
        count1 = 0
        count2 = 0
        for elem in self.correct_y_test:
            count2 += 1
            if elem in self.data[3]:
                count1 += 1
        for elem in self.data[3]:
            count2 += 1
            if elem in self.correct_y_test:
                count1 += 1
        self.assertEqual(count1, count2)

class ExtractFeaturesKmer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.metadata = self.dir + 'TEMPmetadata'
        self.fasta = self.dir + 'TEMP'
        file_header = 'Fasta,Class,Dataset\n'
        file_contents = 'TEMP,A,Train\n'
        with open(self.metadata, 'w') as f:
            f.write('%s%s' % (file_header, file_contents))
        with open(self.fasta, 'w') as f:
            f.write('>label1\nAAAA\n>label2\nCCCC\n>label3\nATAT\n>label4\nACGT')
        kwargs = {'metadata':self.metadata, 'prefix':self.dir, 'validate':False}
        self.data = get_kmer(kwargs,self.db,recount=True,k=4,l=0,extract=True)
        self.correct_features = np.array(['AAAA', 'ACGT', 'ATAT', 'CCCC'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_features(self):
        if np.array_equal(self.correct_features, self.data[1]):
            val = True
        else:
            val = False
        self.assertTrue(val, msg=str(self.data[1]))

class ExtractFeaturesGenomeRegions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.table = self.dir + 'TEMPTable'
        self.metadata = self.dir + 'TEMPMetadata'
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\nTEMP,A,Train')
        with open(self.table, 'w') as f:
            f.write(',TEMP\nA,1\nB,1\nC,0\nD,1\n')
        kwargs = {'metadata': self.metadata, 'validate':False}
        self.data=get_genome_regions(kwargs,table=self.table,sep=None,extract=True)
        self.correct_features = np.array(['A','B','C','D'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_features(self):
        if np.array_equal(self.correct_features, self.data[1]):
            val = True
        else:
            val = False
        self.assertTrue(val)

class ExtractFeaturesOmnilog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.omnilog = self.dir + 'TEMPomnilog'
        self.metadata = self.dir + 'TEMPmetadata'
        with open(self.omnilog, 'w') as f:
            f.write(',A1,A2\na,3.1,3.0\nb,1,1\nc,0.01,0.1\n')
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\nA1,A,Train\nA2,A,Test')
        kwargs = {'metadata': self.metadata, 'validate':True}
        self.data = get_omnilog_data(kwargs,omnilog_sheet=self.omnilog,extract=True)
        self.correct_features = np.array(['a','b','c'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_features(self):
        if np.array_equal(self.correct_features, self.data[1]):
            val = True
        else:
            val = False
        self.assertTrue(val)

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_data.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
