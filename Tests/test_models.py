import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil

# TODO: Write this test
class NeuralNetworkValidation(unittest.TestCase):
    def setUp(self):
        pass

# TODO: Write this test
class NeuralNetwork(unittest.TestCase):
    def setUp(self):
        pass

# TODO: Write this test
class SupportVectorMachineValidation(unittest.TestCase):
    def setUp(self):
        pass

# TODO: Write this test
class SupportVectorMachine(unittest.TestCase):
    def setUp(self):
        pass


if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_models.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
