import unittest
import shutil


# TODO: Write this test
class NeuralNetwork(unittest.TestCase):
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
