import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import constants
import subprocess
import shutil
import tempfile
import yaml
from random import randint
import numpy as np


class CommandLineVariation1(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.reps = 2
        self.samples=10
        self.classes=2
        self.add_samples = 2
        self.train_size=7
        a = ['A','C','G','T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta=[a[randint(0,3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s\n"%fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta=[a[randint(0,3)] for _ in range(500)]
                fasta = ''.join(fasta)
                f.write("%s"%fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,'%i)
                f.write('%d,'%(i%self.classes))
                if i < self.train_size:
                    f.write('Train\n')
                else:
                    f.write('Test\n')
        self.config = {'model': 'support_vector_machine_validation',
                       'model_args': {},
                       'data': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta'},
                                     'database': self.db,
                                     'recount': True,
                                     'k': 3,
                                     'l': 1},
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
            command = constants.SOURCE + 'run.py'
            args = ['python', command, self.config_file, self.results_file]
            self.output = subprocess.check_output(args)
        except subprocess.CalledProcessError:
            self.output = None

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertIsNotNone(self.output)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)
        A = lambda x,y: [x==y, x, y]
        v = {}
        v['reps']=A(data['repetitions'],self.reps)
        v['results_length']=A(len(data['results']),self.reps)
        v['run_times_length']=A(len(data['run_times']),self.reps)
        v['test_size_length']=A(len(data['test_sizes']),self.reps)
        v['train_size_length']=A(len(data['train_sizes']),self.reps)
        v['std_results']=A(data['std_dev_results'],np.asarray(data['results']).std())
        v['avg_results']=A(data['avg_result'],np.asarray(data['results']).mean())
        v['std_time']=A(data['std_dev_run_times'],np.asarray(data['run_times']).std())
        v['avg_time']=A(data['avg_run_time'],np.asarray(data['run_times']).mean())
        v['test_sizes']=A(sum(data['test_sizes']),self.reps*(self.samples-self.train_size))
        v['train_sizes']=A(sum(data['train_sizes']),self.reps*(self.train_size+(self.add_samples*self.classes)))
        vals = [x[0] for x in v.values()]
        val = False if False in vals else True
        self.assertTrue(val, msg={k:v[1:] for k,v in v.items() if not v[0]})


class CommandLineVariation2(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.db = self.dir + 'TEMPDB'
        self.config_file = self.dir + 'config.yml'
        self.fasta_dir = tempfile.mkdtemp() + '/'
        self.metadata = self.dir + 'metadata'
        self.results_file = self.dir + 'results.txt'
        self.samples = 25
        self.classes = 3
        self.add_samples = 3
        self.train_size = 20
        a = ['A','C','G','T']
        for i in range(self.samples):
            with open(self.fasta_dir + str(i) + '.fasta', 'w') as f:
                f.write('>%d.fastaNODE_1\n')
                fasta=[a[randint(0,3)] for _ in range(1500)]
                fasta = ''.join(fasta)
                f.write("%s\n"%fasta)
                f.write('>%d.fastaNODE_2\n')
                fasta=[a[randint(0,3)] for _ in range(1750)]
                fasta = ''.join(fasta)
                f.write("%s"%fasta)
        with open(self.metadata, 'w') as f:
            f.write('Fasta,Class,Dataset\n')
            for i in range(self.samples):
                f.write('%d,'%i)
                if i < self.train_size:
                    f.write('%d,Train\n'%(i%self.classes))
                else:
                    f.write(',Test\n')
        self.config = {'model': 'support_vector_machine',
                       'data': 'get_kmer',
                       'data_args': {'kwargs': {'metadata': self.metadata,
                                                'prefix': self.fasta_dir,
                                                'suffix': '.fasta',
                                                'validate': False},
                                     'database': self.db,
                                     'recount': True,
                                     'k': 4,
                                     'l': 10},
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
            args = ['python', command, self.config_file, self.results_file]
            self.output = subprocess.check_output(args)
        except subprocess.CalledProcessError:
            self.output = None

    def tearDown(self):
        shutil.rmtree(self.dir)
        shutil.rmtree(self.fasta_dir)

    def test_output(self):
        self.assertIsNotNone(self.output)
        with open(self.results_file, 'r') as f:
            data = yaml.load(f)
        A = lambda x,y: [x==y, x, y]
        v = {}
        v['predictions_length']=A(len(data['predictions']),self.samples-self.train_size)
        v['test_sizes']=A(data['test_size'],(self.samples-self.train_size))
        v['train_sizes']=A(data['train_size'],(self.train_size+(self.add_samples*self.classes)))
        vals = [x[0] for x in v.values()]
        val = False if False in vals else True
        self.assertTrue(val, msg={k:v[1:] for k,v in v.items() if not v[0]})


class CommandLineVariation3(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.tempdb = self.dir + '/TEMPdb'
        self.args = ['python',constants.SOURCE+'run.py','-ma','rbf','0.1','-da',
                     self.tempd,'True','3','1','-Sa','0','2','-s',
                     'select_percentile','-sa','chi2','25','-a',
                     'augment_data_naive','-aa','10','--reps','3','-validate',
                     'True','--record_time','True','--record_std_dev','True',
                     '--record_data_size','True']
        self.output = subprocess.check_output(self.args)

    def tearDown(self):
        shutil.rmtree(self.dir)


class FromScript(unittest.TestCase):
    def setUp(self):
        pass

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_run.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
