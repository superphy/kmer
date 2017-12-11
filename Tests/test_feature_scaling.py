import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
import pandas as pd
from feature_scaling import scale_to_range

class ScaleToRange(unittest.TestCase):
    def setUp(self):
        x_train = np.random.randint(10, size=(12,6)).astype('float64')
        y_train = np.random.randint(2, size=12)
        x_test = np.random.randint(10, size=(6,6)).astype('float64')
        y_test = np.random.randint(2, size=6)
        self.low = -1.0
        self.high = 1.0
        self.data = [x_train, y_train, x_test, y_test]
        self.new_data = scale_to_range(self.data, low=self.low, high=self.high)

    def test_labels(self):
        val1 = np.array_equal(self.data[1], self.new_data[1])
        val2 = np.array_equal(self.data[3], self.new_data[3])
        self.assertTrue(val1 and val2)

    def test_range(self):
        func = np.vectorize(lambda x: True if (x<=self.high and x>=self.low) else False)
        a = func(self.new_data[0])
        self.assertTrue(a.all(), msg=str(a)+'\n'+str(self.new_data[0])+'\nShould all be <= %d and >= to %d'%(self.high,self.low))

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_feature_scaling.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
