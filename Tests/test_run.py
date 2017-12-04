import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil


class FromCommandLine(unittest.TestCase):
    def setUp(self):
        pass

class FromScript(unittest.TestCase):
    def setUp(self):
        pass

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_run.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
