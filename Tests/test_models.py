import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil


class NeuralNetworkValidation(unittest.TestCase):
    def setUp(self):
        pass


class NeuralNetwork(unittest.TestCase):
    def setUp(self):
        pass


class SupportVectorMachineValidation(unittest.TestCase):
    def setUp(self):
        pass


class SupportVectorMachine(unittest.TestCase):
    def setUp(self):
        pass


if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_models.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
