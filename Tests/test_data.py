import sys
import os
import path
sys.path.append(os.path.join(os.path.dirname(__file__), "../kmerprediction"))
from builtins import str
import os
import unittest
import shutil
from get_data import get_kmer, get_genome_regions, get_omnilog_data
from get_data import get_genome_custom_filtered, get_genome_prefiltered
from get_data import get_kmer_from_directory, get_kmer_from_json
import numpy as np
import tempfile
import json


class GetKmer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = self.dir + '/TEMPdatabase'
        self.metadata = self.dir + '/TEMPmetadata'
        file_header = 'Fasta,Class,Dataset\n'
        file_contents = 'A1,A,Train\nA2,A,Test\nB1,B,Train'
        with open(self.metadata, 'w') as f:
            f.write('%s%s' % (file_header, file_contents))
        with open(self.dir + '/A1', 'w') as f:
            f.write('>\nAAACCCCAA')
        with open(self.dir + '/A2', 'w') as f:
            f.write('>\nAACCCCAA')
        with open(self.dir + '/B1', 'w') as f:
            f.write('>\nAACCAACC')
        kwargs = {'metadata': self.metadata, 'prefix': self.dir + '/'}
        self.data, self.features, self.files, self.le = get_kmer(kwargs,
                                                                 self.db,
                                                                 recount=True,
                                                                 k=2, L=1,
                                                                 verbose=False)
        self.correct_x_train = np.array([[3, 1, 1, 3], [2, 2, 1, 2]])
        self.correct_y_train = np.array(['A', 'B'])
        self.correct_x_test = np.array([[2, 1, 1, 3]])
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
            if elem in self.le.inverse_transform(self.data[1]):
                count1 += 1
        for elem in self.le.inverse_transform(self.data[1]):
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
        if np.array_equal(self.le.inverse_transform(self.data[3]),
                          self.correct_y_test):
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
            f.write('%s%s' % (file_header, file_contents))
        with open(self.table, 'w') as f:
            f.write('%s%s' % (table_header, table_contents))
        kwargs = {'metadata': self.metadata}
        self.data, fea, files, self.le = get_genome_regions(kwargs,
                                                            table=self.table,
                                                            sep=None)
        self.correct_x_train = np.array([[1, 1, 0, 1, 0, 1],
                                         [0, 1, 1, 1, 0, 1]])
        self.correct_y_train = np.array(['A', 'B'])
        self.correct_x_test = np.array([[1, 0, 0, 1, 1, 0]])
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
            if elem in self.le.inverse_transform(self.data[1]):
                count1 += 1
        for elem in self.le.inverse_transform(self.data[1]):
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
        if np.array_equal(self.le.inverse_transform(self.data[3]),
                          self.correct_y_test):
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
            out = 'A1,A,Train\nA2,A,Train\nA3,A,Train\nA4,A,Train\nA5,A,Test\n'
            f.write(out)
            out = 'B1,B,Train\nB2,B,Train\nB3,B,Train\nB4,B,Test\nB5,B,Test\n'
            f.write(out)
        kwargs = {'metadata': self.metadata}
        self.data, f, fi, self.L = get_omnilog_data(kwargs,
                                                    omnilog_sheet=self.omnilog)
        self.correct_x_train = np.array([[3.1, 1, 0.01], [3.4, 1, 0.3],
                                         [2.43, 1, 0.1], [4.2, 0, 0.56],
                                         [3.3, 0, 0.2], [3.53, 1, 0.34],
                                         [2.7, 1, 0.50]])
        self.correct_y_train = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B'])
        self.correct_x_test = np.array([[3.3, 0, 0.23], [2.8, 1, 0.02],
                                        [4.1, 1, 0.34]])
        self.correct_y_test = np.array(['A', 'B', 'B'])

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
            if elem in self.L.inverse_transform(self.data[1]):
                count1 += 1
        for elem in self.L.inverse_transform(self.data[1]):
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
            if elem in self.L.inverse_transform(self.data[3]):
                count1 += 1
        for elem in self.L.inverse_transform(self.data[3]):
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
            out = '>label1\nAAAA\n>label2\nCCCC\n>label3\nATAT\n>label4\nACGT'
            f.write(out)
        kwargs = {'metadata': self.metadata, 'prefix': self.dir,
                  'validate': False}
        self.data = get_kmer(kwargs, self.db, recount=True, k=4, L=0,
                             validate=False, verbose=False)
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
        kwargs = {'metadata': self.metadata, 'validate': False}
        self.data = get_genome_regions(kwargs, table=self.table, sep=None,
                                       validate=False)
        self.correct_features = np.array(['A', 'B', 'C', 'D'])

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
        kwargs = {'metadata': self.metadata, 'validate': True}
        self.data = get_omnilog_data(kwargs, omnilog_sheet=self.omnilog)
        self.correct_features = np.array(['a', 'b', 'c'])

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_features(self):
        if np.array_equal(self.correct_features, self.data[1]):
            val = True
        else:
            val = False
        self.assertTrue(val)


class GenomeFiltered(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.input_table = self.dir + 'input_table'
        self.filter_table = self.dir + 'filter_table'
        self.metadata = self.dir + 'metadata'
        self.kwargs = {'metadata': self.metadata}
        self.args = {'input_table': self.input_table,
                     'filter_table': self.filter_table,
                     'kwargs': self.kwargs, 'sep': None}
        with open(self.input_table, 'w') as f:
            out = ',A1,A2,B1\n'
            f.write(out)
            out = 'a,1,1,0\nb,1,0,1\nc,0,0,1\nd,1,1,1\ne,0,1,0\nf,1,0,1'
            f.write(out)
        with open(self.filter_table, 'w') as f:
            out = ',Garbag2,Correct,Garbage1\n'
            f.write(out)
            out = 'a,12,1,abc\nc,1,0.975,gef\ne,2,-0.75,hij\nd,13,-0.65,jkl\n'
            f.write(out)
            out = 'f,34,0.34,rst\nb,-12,0.12,uvx\n'
            f.write(out)
        with open(self.metadata, 'w') as f:
            out = 'Fasta,Class,Dataset\n'
            f.write(out)
            out = 'A1,A,Train\nA2,A,Test\nB1,B,Train'
            f.write(out)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_custom_filter(self):
        self.args['absolute'] = False
        self.args['greater'] = False
        self.args['cutoff'] = 0.3
        self.args['col'] = 'Correct'
        data, fe_n, fi_n, le = get_genome_custom_filtered(**self.args)
        correct_fe_n = ['e', 'd', 'b']
        count1 = 0
        count2 = 0
        for i in fe_n:
            count1 += 1
            if i in correct_fe_n:
                count2 += 1
        for i in correct_fe_n:
            count1 += 1
            if i in fe_n:
                count2 += 1
        self.assertEqual(count1, count2)

    def test_prefilter(self):
        self.args['count'] = 3
        data, fe_n, fi_n, le = get_genome_prefiltered(**self.args)
        correct_fe_n = ['a', 'c', 'e']
        count1 = 0
        count2 = 0
        for i in fe_n:
            count1 += 1
            if i in correct_fe_n:
                count2 += 1
        for i in correct_fe_n:
            count1 += 1
            if i in fe_n:
                count2 += 1
        self.assertEqual(count1, count2)


class Directory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'db'
        self.train = tempfile.mkdtemp() + '/'
        self.test = tempfile.mkdtemp() + '/'
        os.makedirs(self.train + 'Class1')
        os.makedirs(self.train + 'Class2')
        os.makedirs(self.train + 'Class3')
        os.makedirs(self.test + 'Class1')
        os.makedirs(self.test + 'Class2')
        self.args = {'train_dir': self.train, 'test_dir': self.test,
                     'database': self.db, 'recount': True, 'k': 3, 'L': 1,
                     'verbose': False}
        with open(self.train + 'Class1/one', 'w') as f:
            f.write('>\nAAACCCCAA')
        with open(self.train + 'Class1/two', 'w') as f:
            f.write('>\nACAAAACCA')
        with open(self.train + 'Class2/three', 'w') as f:
            f.write('>\nAAAAAAACC')
        with open(self.train + 'Class3/four', 'w') as f:
            f.write('>\nCCCCACCAC')
        with open(self.test + 'Class1/five', 'w') as f:
            f.write('>\nCCCCACCCC')
        with open(self.test + 'Class1/six', 'w') as f:
            f.write('>\nAAACCAACC')
        with open(self.test + 'Class2/seven', 'w') as f:
            f.write('>\nAAACAAACA')
        self.cor_fi = [self.test + 'Class1/five', self.test + 'Class1/six',
                       self.test + 'Class2/seven']
        self.cor_y_tr = ['Class1', 'Class1', 'Class2', 'Class3']
        self.cor_y_ts = ['Class1', 'Class1', 'Class2']

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.train)
        shutil.rmtree(self.test)

    def test_validate(self):
        self.args['validate'] = True
        data, fe, fi, le = get_kmer_from_directory(**self.args)
        count1 = 0
        count2 = 0
        for i in fi:
            count1 += 1
            if i in self.cor_fi:
                count2 += 1
        for i in self.cor_fi:
            count1 += 1
            if i in fi:
                count2 += 1
        for i in le.transform(self.cor_y_tr):
            count1 += 1
            if i in list(data[1]):
                count2 += 1
        for i in data[1]:
            count1 += 1
            if i in le.transform(self.cor_y_tr):
                count2 += 1
        for i in le.transform(self.cor_y_ts):
            count1 += 1
            if i in list(data[3]):
                count2 += 1
        for i in data[3]:
            count1 += 1
            if i in le.transform(self.cor_y_ts):
                count2 += 1
        self.assertEqual(count1, count2)

    def test_non_validate(self):
        self.args['validate'] = False
        data, fe, fi, le = get_kmer_from_directory(**self.args)
        count1 = 0
        count2 = 0
        for i in fi:
            count1 += 1
            if i in self.cor_fi:
                count2 += 1
        for i in self.cor_fi:
            count1 += 1
            if i in fi:
                count2 += 1
        for i in le.transform(self.cor_y_tr):
            count1 += 1
            if i in list(data[1]):
                count2 += 1
        for i in data[1]:
            count1 += 1
            if i in le.transform(self.cor_y_tr):
                count2 += 1
        if data[3].shape != (0,):
            count1 += 1
        self.assertEqual(count1, count2)


class GetKmerFromJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'db'
        self.json1 = self.dir + 'json1'
        self.json2 = self.dir + 'json2'
        with open(self.json1, 'w') as f:
            f.write(json.dumps([{"class": 1, "name": 'one', 'garbage': 456},
                                {"class": 1, "name": 'two', 'garbage': 654},
                                {'class': 2, "name": 'three', 'garbage': 45},
                                {'class': 3, "name": 'four', 'garbage': -12}]))
        with open(self.json2, 'w') as f:
            f.write(json.dumps([{"class": 1, "name": 'five', 'garbage': 456},
                                {"class": 1, "name": 'six', 'garbage': 654},
                                {'class': 2, "name": 'seven', 'garbage': 4}]))
        with open(self.dir + 'one', 'w') as f:
            f.write('>\nAAACCCCAA')
        with open(self.dir + 'two', 'w') as f:
            f.write('>\nACAAAACCA')
        with open(self.dir + 'three', 'w') as f:
            f.write('>\nAAAAAAACC')
        with open(self.dir + 'four', 'w') as f:
            f.write('>\nCCCCACCAC')
        with open(self.dir + 'five', 'w') as f:
            f.write('>\nCCCCACCCC')
        with open(self.dir + 'six', 'w') as f:
            f.write('>\nAAACCAACC')
        with open(self.dir + 'seven', 'w') as f:
            f.write('>\nAAACCAACA')
        kwargs = {'prefix': self.dir, 'suffix': '', 'fasta_key': "name",
                  'label_key': "class"}
        self.data, fe, self.fi, self.le = get_kmer_from_json(self.json1,
                                                             self.json2,
                                                             recount=True,
                                                             k=3, L=1,
                                                             kwargs=kwargs,
                                                             verbose=False,
                                                             validate=False)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_file_names(self):
        correct_file_names = [self.dir + 'five', self.dir + 'six',
                              self.dir + 'seven']
        count1 = 0
        count2 = 0
        for i in self.fi:
            count1 += 1
            if i in correct_file_names:
                count2 += 1
        for i in correct_file_names:
            count1 += 1
            if i in self.fi:
                count2 += 1
        self.assertEqual(count1, count2)

    def test_y_train(self):
        correct_y_train = self.le.transform(['1', '1', '2', '3'])
        count1 = 0
        count2 = 0
        for i in correct_y_train:
            count1 += 1
            if i in list(self.data[1]):
                count2 += 1
        for i in list(self.data[1]):
            count1 += 1
            if i in correct_y_train:
                count2 += 1
        self.assertEqual(count1, count2)

    def test_y_tes(self):
        val = bool(self.data[3].shape == (0,))
        self.assertTrue(val)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_data.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
