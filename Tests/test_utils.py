import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import random
import numpy as np
import pandas as pd
import string
from utils import same_shuffle, shuffle, parse_metadata


class SameShuffle(unittest.TestCase):
    def setUp(self):
        self.length = 15
        self.a = [x for x in range(self.length)]
        self.b = [x for x in range(self.length)]
        self.new_a, self.new_b = same_shuffle(self.a, self.b)

    def test_order(self):
        val1 = np.array_equal(self.new_a, self.new_b)
        val2 = np.array_equal(self.new_a, self.a)
        val3 = np.array_equal(self.new_b, self.b)
        self.assertTrue(val1 and (not val2) and (not val3))

    def test_dimensions(self):
        val1 = np.array_equal(len(self.new_a), len(self.a))
        val2 = np.array_equal(len(self.new_b), len(self.b))
        self.assertTrue(val1 and val2)


class ShuffleArray(unittest.TestCase):
    def setUp(self):
        self.max_samples = 10
        self.features = 7
        self.classes = 12
        self.samples_per_class = [random.randint(1, self.max_samples) for _ in range(self.classes)]
        self.data = np.zeros(self.classes, dtype=object)
        for x in range(self.classes):
            self.data[x] = np.full((self.samples_per_class[x], self.features), x)
        self.labels = np.arange(self.classes)
        self.new_data, self.new_labels = shuffle(self.data, self.labels)

    def test_order(self):
        count1 = 0
        count2 = 0
        for elem in range(self.new_data.shape[0]):
            count2 += 1
            if self.new_data[elem][0] == self.new_labels[elem]:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_data_dimensions(self):
        val = False
        if self.new_data.shape == (sum(self.samples_per_class), self.features):
            val = True
        self.assertTrue(val)

    def test_label_dimensions(self):
        val = False
        if self.new_labels.shape == (sum(self.samples_per_class),):
            val = True
        self.assertTrue(val)


class ShuffleList(unittest.TestCase):
    def setUp(self):
        self.max_samples = 10
        self.features = 7
        self.classes = 12
        self.samples_per_class = [random.randint(1, self.max_samples) for _ in range(self.classes)]
        self.data = [[[x for y in range(self.features)] for z in range(self.samples_per_class[x])] for x in range(self.classes)]
        self.labels = [x for x in range(self.classes)]
        self.new_data, self.new_labels = shuffle(self.data, self.labels)

    def test_order(self):
        count1 = 0
        count2 = 0
        for elem in range(len(self.new_labels)):
            count2 += 1
            if self.new_data[elem][0] == self.new_labels[elem]:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_data_dimensions(self):
        val = False
        if len(self.new_data) == sum(self.samples_per_class):
            val = True
        self.assertTrue(val)

    def test_label_dimensions(self):
        val = False
        if len(self.new_labels) == sum(self.samples_per_class):
            val = True
        self.assertTrue(val)


class ParseMetadata(unittest.TestCase):
    def setUp(self):
        self.file = 'metadata.temp'
        self.length = 100
        self.classes = 3
        self.classifications = list(string.ascii_lowercase)
        self.datasets = ['Train', 'Test']
        data = {'Filename':[x for x in range(self.length)],
                'Classification':[self.classifications[x%self.classes] for x in range(self.length)],
                'Dataset':[self.datasets[x%len(self.datasets)] for x in range(self.length)],
                'Extra':[random.uniform(0,1) for _ in range(self.length)],
                'Ignore':[random.randint(0,25) for _ in range(self.length)]}
        self.data = pd.DataFrame(data=data)
        self.data.to_csv(self.file)

    def tearDown(self):
        os.remove(self.file)

    def test_default_x_train(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        correct = [str(x) for x in range(self.length) if x%2 == 0]
        val = False
        if np.array_equal(np.unique(x_train, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_default_x_test(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        correct = [str(x) for x in range(self.length) if x%2 == 1]
        val = False
        if np.array_equal(np.unique(x_test, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_default_y_train(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        val = False
        if np.array_equal(np.unique(y_train, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_default_y_test(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        val = False
        if np.array_equal(np.unique(y_test, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_default_values(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        count1 = 0
        count2 = 0
        correct_x = [str(x) for x in range(self.length) if x%2 == 0]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        for elem in x_train:
            index = correct_x.index(elem)
            if y_train[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

    def test_default_labels(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file)
        count1 = 0
        count2 = 0
        correct_x = [str(x) for x in range(self.length) if x%2 == 1]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        for elem in x_test:
            index = correct_x.index(elem)
            if y_test[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_utils.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
