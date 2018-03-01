from builtins import str
from builtins import range
import unittest
from kmerprediction import constants
import subprocess
import shutil
import tempfile
import yaml
from random import randint
import numpy as np
from kmerprediction.run import main


class CommandLineVariation1(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.reps = 2
        self.samples = 10
        self.classes = 2
        self.add_samples = 2
        self.train_size = 7
        a = ['A', 'C', 'G', 'T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta = [a[randint(0, 3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s\n" % fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta = [a[randint(0, 3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s" % fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,' % i)
                f.write('%d,' % (i % self.classes))
                if i < self.train_size:
                    f.write('Train\n')
                else:
                    f.write('Test\n')
        self.config = {'model': 'support_vector_machine',
                       'model_args': {'validate': True},
                       'data_method': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta'},
                                     'database': self.db,
                                     'recount': True,
                                     'k': 3,
                                     'L': 1},
                       'scaler': 'scale_to_range',
                       'scaler_args': {'low': -2,
                                       'high': 2},
                       'selection': 'select_k_best',
                       'selection_args': {'score_func': 'f_classif',
                                          'k': 15},
                       'augment': 'augment_data_noise',
                       'augment_args': {'desired_samples': self.add_samples},
                       'validate': True,
                       'reps': self.reps}
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        try:
            self.exception_message = ''
            command = constants.SOURCE + 'run.py'
            args = ['python', command, '-i', self.config_file, '-o',
                    self.results_file]
            self.output = subprocess.check_output(args)
        except subprocess.CalledProcessError as e:
            self.output = None
            self.esception_message = e

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertIsNotNone(self.output, msg=self.exception_message)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)

        def A(x, y):
            return [x == y, x, y]
        v = {}
        v['reps'] = A(data['output']['repetitions'], self.reps)
        v['results_length'] = A(len(data['output']['results']), self.reps)
        v['std_results'] = A(data['output']['std_dev_results'], np.asarray(data['output']['results']).std())
        v['avg_results'] = A(data['output']['avg_result'], np.asarray(data['output']['results']).mean())
        v['test_sizes'] = A(data['output']['test_sizes'], (self.samples - self.train_size))
        v['train_sizes'] = A(data['output']['train_sizes'], (self.train_size + (self.add_samples * self.classes)))
        vals = [x[0] for x in list(v.values())]
        val = False if False in vals else True
        self.assertTrue(val, msg={k: v[1:] for k, v in list(v.items()) if not v[0]})


class CommandLineVariation2(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.samples = 25
        self.classes = 2
        self.add_samples = 3
        self.train_size = 20
        a = ['A', 'C', 'G', 'T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta = [a[randint(0, 3)] for _ in range(1500)]
                fasta = ''.join(fasta)
                f.write("%s\n" % fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta = [a[randint(0, 3)] for _ in range(1750)]
                fasta = ''.join(fasta)
                f.write("%s" % fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,' % i)
                if i < self.train_size:
                    f.write('%d,Train\n' % (i % self.classes))
                else:
                    f.write(',Test\n')
        self.config = {'model': 'support_vector_machine',
                       'model_args': {'validate': False},
                       'data_method': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta',
                                                'validate': False,
                                                'train_header': 'Dataset',
                                                'train_label': 'Train',
                                                'test_label': 'Test'},
                                     'database': self.db,
                                     'recount': True,
                                     'k': 4,
                                     'L': 2},
                       'selection': 'select_percentile',
                       'selection_args': {'score_func': 'chi2',
                                          'percentile': 0.5},
                       'augment': 'augment_data_naive',
                       'augment_args': {'desired_samples': self.add_samples},
                       'validate': False}
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        try:
            command = constants.SOURCE + 'run.py'
            args = ['python', command, '-i', self.config_file, '-o',
                    self.results_file]
            self.output = subprocess.check_output(args)
            self.exception_message = ''
        except subprocess.CalledProcessError as e:
            self.output = None
            self.exception_message = e

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertIsNotNone(self.output, msg=self.exception_message)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)

        def A(x, y):
            return [x == y, x, y]
        v = {}
        v['results_length'] = A(len(data['output']['results']),
                                self.samples - self.train_size)
        v['results_type'] = A(type(data['output']['results']), dict)
        v['test_sizes'] = A(data['output']['test_sizes'], (self.samples - self.train_size))
        v['train_sizes'] = A(data['output']['train_sizes'],  (self.train_size + (self.add_samples * self.classes)))
        vals = [x[0] for x in list(v.values())]
        val = False if False in vals else True
        self.assertTrue(val, msg={k: v[1:] for k, v in list(v.items()) if not v[0]})


class ScriptVariation1(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.samples = 25
        self.classes = 6
        self.train_size = 20
        a = ['A', 'C', 'G', 'T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta = [a[randint(0, 3)] for _ in range(6000)]
                fasta = ''.join(fasta)
                f.write("%s\n" % fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta = [a[randint(0, 3)] for _ in range(7000)]
                fasta = ''.join(fasta)
                f.write("%s" % fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,' % i)
                if i < self.train_size:
                    f.write('%d,Train\n' % (i % self.classes))
                else:
                    f.write(',Test\n')
        self.config = {'model': 'random_forest',
                       'model_args': {'validate': False},
                       'data_method': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta',
                                                'validate': True,
                                                'train_header': 'Dataset',
                                                'train_label': 'Train',
                                                'test_label': 'Test'},
                                     'database': self.db,
                                     'recount': True,
                                     'verbose': False,
                                     'k': 5,
                                     'L': 1},
                       'validate': False}
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        self.output = True
        try:
            main(self.config_file, self.results_file, '')
            self.exception_message = ''
        except Exception as e:
            self.output = None
            self.exception_message = e

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertIsNotNone(self.output, msg=self.exception_message)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)

        def A(x, y):
            return [x == y, x, y]
        v = {}
        v['results_length'] = A(len(data['output']['results']),
                                self.samples - self.train_size)
        v['results_type'] = A(type(data['output']['results']), dict)
        v['test_sizes'] = A(data['output']['test_sizes'], (self.samples - self.train_size))
        v['train_sizes'] = A(data['output']['train_sizes'], (self.train_size))
        vals = [x[0] for x in list(v.values())]
        val = False if False in vals else True
        self.assertTrue(val, msg={k: v[1:] for k, v in list(v.items()) if not v[0]})


class ScriptVariation2(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.reps = 1
        self.samples = 30
        self.classes = 3
        self.add_samples = 4
        self.train_size = 15
        a = ['A', 'C', 'G', 'T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta = [a[randint(0, 3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s\n" % fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta = [a[randint(0, 3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s" % fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,' % i)
                f.write('%d,' % (i % self.classes))
                if i < self.train_size:
                    f.write('Train\n')
                else:
                    f.write('Test\n')
        self.config = {'model': 'support_vector_machine',
                       'model_args': {'validate': True},
                       'data_method': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta'},
                                     'database': self.db,
                                     'recount': True,
                                     'verbose': False,
                                     'k': 4,
                                     'L': 1},
                       'scaler': 'scale_to_range',
                       'scaler_args': {'low': 0,
                                       'high': 4},
                       'selection': 'select_k_best',
                       'selection_args': {'score_func': 'f_classif',
                                          'k': 15},
                       'augment': 'augment_data_noise',
                       'augment_args': {'desired_samples': self.add_samples},
                       'validate': True,
                       'reps': self.reps}
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        try:
            main(self.config_file, self.results_file, 'Run1')
            self.exception_message = ''
        except Exception as e:
            self.exception_message = e

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertEqual(self.exception_message, '', msg=self.exception_message)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)

        def A(x, y):
            return [x == y, x, y]
        v = {}
        v['reps'] = A(data['output']['repetitions'], self.reps)
        v['results_length'] = A(len(data['output']['results']), self.reps)
        v['std_results'] = A(data['output']['std_dev_results'], np.asarray(data['output']['results']).std())
        v['avg_results'] = A(data['output']['avg_result'], np.asarray(data['output']['results']).mean())
        v['test_sizes'] = A(data['output']['test_sizes'], (self.samples - self.train_size))
        v['train_sizes'] = A(data['output']['train_sizes'], (self.train_size + (self.add_samples * self.classes)))
        vals = [x[0] for x in list(v.values())]
        val = False if False in vals else True
        self.assertTrue(val, msg={k: v[1:] for k, v in list(v.items()) if not v[0]})


if __name__ == "__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_run.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
