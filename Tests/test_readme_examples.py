import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../kmerprediction"))
import unittest
import tempfile
import shutil
import yaml


class FirstExample(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.config = {'model': 'neural_network',
                       'data_method': 'get_kmer_us_uk_split',
                       'validate': False}
        self.config_file = self.dir + 'config_file.yml'
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        self.output_file = self.dir + 'output_file.yml'

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_import(self):
        error_msg = ''
        try:
            from run import main
        except ImportError as e:
            error_msg = e
        self.assertEqual(error_msg, '', msg=error_msg)

    def test_main(self):
        error_msg = ''
        try:
            from run import main
            main(self.config_file, self.output_file, 'name of run')
        except Exception as e:
            error_msg = ''
        self.assertEqual(error_msg, '', msg=error_msg)


class SecondExample(unittest.TestCase):
    def test_import(self):
        error_msg = ''
        try:
            from run import run
        except ImportError as e:
            error_msg = e
        self.assertEqual(error_msg, '', msg=error_msg)

    def test_run(self):
        error_msg = ''
        try:
            from run import run
            output = run()
        except Exception as e:
            error_msg = ''
        self.assertEqual(error_msg, '', msg=error_msg)


class ThirdExample(unittest.TestCase):
    def test_import(self):
        error_msg = ''
        try:
            from models import neural_network as nn
            from get_data import get_genome_region_us_uk_split as data
            from feature_selection import variance_threshold as sel
            from run import run
        except ImportError as e:
            error_msg = e
        self.assertEqual(error_msg, '', msg=error_msg)

    def test_run(self):
        error_msg = ''
        try:
            from models import neural_network as nn
            from get_data import get_genome_region_us_uk_split as data
            from feature_selection import variance_threshold as sel
            from run import run
            output = run(model=nn, data_method=data, selection=sel,
                         selection_args={'threshold': 0.01}, scaler=None,
                         reps=1, validate=True)
        except Exception as e:
            error_msg = ''
        self.assertEqual(error_msg, '', msg=error_msg)


class FourthExample(unittest.TestCase):
    def test_import(self):
        error_msg = ''
        try:
            from models import neural_network
            from get_data import get_genome_region_us_uk_mixed as data
            from feature_selection import variance_threshold as sel
        except ImportError as e:
            error_msg = e
        self.assertEqual(error_msg, '', msg=error_msg)

    def test_run(self):
        error_msg = ''
        try:
            from models import neural_network
            from get_data import get_genome_region_us_uk_mixed as data
            from feature_selection import variance_threshold as sel
            d = data()
            d, f = sel(d[0], d[1], threshold=0.01)
            score = neural_network(d)
        except Exception as e:
            error_msg = e
        self.assertEqual(error_msg, '', msg=error_msg)
