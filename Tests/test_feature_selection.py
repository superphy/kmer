import sys
sys.path.append('/home/rboothman/PHAC/kmer/')
import unittest
import numpy as np
from feature_selection import variance_threshold, remove_constant_features
from feature_selection import select_k_best, select_percentile
from feature_selection import recursive_feature_elimination
from feature_selection import recursive_feature_elimination_cv

class BinaryBase(unitttest.TestCase):
    def __init__(self, testName, method, args):
        super(BinaryBase, self).__init__(testName)
        self.method = method
        self.args = args
        
