from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import unittest
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from feature_selection import variance_threshold, remove_constant
from feature_selection import select_k_best, select_percentile


class VarianceThreshold(unittest.TestCase):
    def setUp(self):
        self.classes = 3
        x_train = np.random.randint(15, size=(12, 6))
        y_train = np.random.randint(self.classes, size=(12))
        x_test = np.random.randint(15, size=(12, 6))
        y_test = np.random.randint(self.classes, size=(12))
        self.data = [x_train, y_train, x_test, y_test]
        self.threshold = 0.75 * x_train[0][0].var()
        train_df = pd.DataFrame(x_train)
        test_df = pd.DataFrame(x_test)
        train_df = train_df.drop(train_df.var()[train_df.var() <
                                 self.threshold].index.values, axis=1)
        test_df = test_df[train_df.columns]
        new_x_train = train_df.values
        new_x_test = test_df.values
        self.correct_data = [new_x_train, y_train, new_x_test, y_test]
        self.new_data, self.fn = variance_threshold(self.data, None,
                                                    self.threshold)

    def test_dimensions(self):
        count1 = 0
        count2 = 0
        for x in range(len(self.correct_data)):
            count2 += 1
            if self.new_data[x].shape == self.correct_data[x].shape:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_values(self):
        count1 = 0
        count2 = 0
        for x in range(len(self.correct_data)):
            count2 += 1
            if np.array_equal(self.new_data[x], self.correct_data[x]):
                count1 += 1
        self.assertEqual(count1, count2)

    def test_variances(self):
        count1 = 0
        count2 = 0
        for elem in np.var(self.new_data[0], axis=0):
            count2 += 1
            if elem >= self.threshold:
                count1 += 1
        for elem in np.var(self.new_data[2], axis=0):
            count2 += 1
            if elem >= self.threshold:
                count1 += 1
        self.assertEqual(count1, count2)


class RemoveConstantFeatures(unittest.TestCase):
    def setUp(self):
        a = np.zeros((12, 1))
        b = np.random.randint(15, size=(12, 4))
        self.x_train = np.hstack((a, b, a))
        self.x_test = np.random.randint(15, size=(6, 6))
        self.data = (self.x_train, [], self.x_test, [])
        self.new_data, self.feat_name = remove_constant(self.data, None)
        self.correct_x_train = b
        self.correct_x_test = np.delete(self.x_test, [0, 5], axis=1)

    def test_values(self):
        val1 = np.array_equal(self.new_data[0], self.correct_x_train)
        val2 = np.array_equal(self.new_data[2], self.correct_x_test)
        self.assertTrue(val1 and val2, msg='\n' + str(self.new_data[2]) + '\n'
                        + str(self.correct_x_test))

    def test_feature_extraction(self):
        features_before = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        features_after = np.array(['b', 'c', 'd', 'e'])
        data, features = remove_constant(self.data, features_before)
        if np.array_equal(features_after, features):
            val = True
        else:
            val = False
        self.assertTrue(val)


class SelectKBest(unittest.TestCase):
    def setUp(self):
        self.score_func = chi2
        self.classes = 3
        self.features = 1000
        best = np.zeros((self.features * self.classes, 1))
        for x in range(self.classes):
            best[x * self.features: (x + 1) * self.features] = x
        c = np.random.randint(2, size=(self.features * self.classes,
                              self.features - 2))
        x_train = np.hstack((best, c, best))
        y_train = best.reshape(self.features * self.classes)
        x_test = np.random.randint(15, size=(10, self.features))
        y_test = np.random.randint(self.classes, size=(10))
        self.data = [x_train, y_train, x_test, y_test]
        self.new_data, self.fn = select_k_best(self.data, None,
                                               score_func=self.score_func, k=2)
        correct_x_train = np.delete(x_train, np.arange(1, self.features - 1),
                                    axis=1)
        correct_x_test = np.delete(x_test, np.arange(1, self.features - 1),
                                   axis=1)
        self.correct_data = [correct_x_train, y_train, correct_x_test, y_test]

    def test_values(self):
        count1 = 0
        count2 = 0
        for x in range(len(self.correct_data)):
            count2 += 1
            if np.array_equal(self.new_data[x], self.correct_data[x]):
                count1 += 1
        self.assertEqual(count1, count2)

    def test_feature_extraction(self):
        features_before = np.random.randint(self.features, size=self.features)
        features_after = features_before[[0, -1]]
        data, features = select_k_best(self.data, features_before,
                                       score_func=self.score_func, k=2)
        if np.array_equal(features_after, features):
            val = True
        else:
            val = False
        self.assertTrue(val)


class SelectPercentile(unittest.TestCase):
    def setUp(self):
        self.score_func = chi2
        self.classes = 3
        self.features = 5
        best = np.zeros((self.features * self.classes, 1))
        for x in range(self.classes):
            best[x * self.features: (x + 1) * self.features] = x
        c = np.random.randint(2, size=(self.features * self.classes,
                              self.features - 2))
        x_train = np.hstack((best, c, best))
        y_train = best.reshape(self.features * self.classes)
        x_test = np.random.randint(15, size=(10, self.features))
        y_test = np.random.randint(self.classes, size=(10))
        self.data = [x_train, y_train, x_test, y_test]
        p = old_div(200, self.features)
        self.new_data, self.f = select_percentile(self.data, None,
                                                  score_func=self.score_func,
                                                  percentile=p)
        correct_x_train = np.delete(x_train, np.arange(1, self.features - 1),
                                    axis=1)
        correct_x_test = np.delete(x_test, np.arange(1, self.features - 1),
                                   axis=1)
        self.correct_data = [correct_x_train, y_train, correct_x_test, y_test]

    def test_values(self):
        count1 = 0
        count2 = 0
        for x in range(len(self.correct_data)):
            count2 += 1
            if np.array_equal(self.new_data[x], self.correct_data[x]):
                count1 += 1
        self.assertEqual(count1, count2, msg=str(self.score_func) + '\n' +
                         str(self.classes))

    def test_feature_extraction(self):
        features_before = np.random.randint(self.features,
                                            size=(self.features))
        features_after = features_before[[0, -1]]
        p = old_div(200, self.features)
        sf = self.score_func
        fn = features_before
        data, features = select_percentile(self.data, fn, score_func=sf,
                                           percentile=p)
        if np.array_equal(features_after, features):
            val = True
        else:
            val = False
        self.assertTrue(val)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_feature_selection.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
