import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil
from data import get_kmer, get_genome_region
import numpy as np
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
        self.data = get_genome_region(kwargs, table=self.table, sep=None)
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

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_data.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
