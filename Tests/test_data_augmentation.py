import sys
sys.path.append('/home/rboothman/PHAC/kmer/')
import unittest
import numpy as np
from data_augmentation import augment_data_naive,augment_data_smote
from data_augmentation import augment_data_adasyn,augment_data_noise


class BinaryBase(unittest.TestCase):
    def __init__(self, testName, method):
        super(BinaryBase, self).__init__(testName)
        self.method = method
        x_train = np.array([[1,1,1,1,2,2],
                            [1,1,1,2,1,2],
                            [2,2,1,1,1,2],
                            [1,2,1,1,1,1],
                            [1,1,1,1,2,1],
                            [1,1,1,2,1,1],
                            [2,2,2,2,1,1],
                            [2,2,2,1,2,1],
                            [2,2,2,1,2,2],
                            [1,2,2,2,2,2],
                            [2,1,2,1,2,2],
                            [2,2,2,2,2,1]])
        y_train = np.array([1,1,1,1,1,1,2,2,2,2,2,2])
        x_test = np.array([[1,1,1,1,1,1],
                           [2,2,2,2,2,2]])
        y_test = np.array([1,2])
        self.data = [x_train, y_train, x_test, y_test]
        self.new_y_train = np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2])

    def test_dimensions(self):
        new_data = self.method(self.data, 6)
        correct_dims = [(24,6), (24,), (2,6), (2,)]
        new_dims = [x.shape for x in new_data]
        count = 0
        for x in range(len(new_dims)):
            if new_dims[x] == correct_dims[x]:
                count+=1
        self.assertEqual(count, 4)

    def test_train_labels(self):
        new_data = self.method(self.data, 6)
        count = 0
        for x in new_data[1].tolist():
            if x in self.new_y_train.tolist():
                count += 1
        for x in self.new_y_train.tolist():
            if x in new_data[1].tolist():
                count += 1
        self.assertEqual(count, 48)

    def test_test_values_and_labels(self):
        new_data = self.method(self.data, 6)
        count = 0
        for x in [-1,-2]:
            if (new_data[x] == self.data[x]).all():
                count += 1
        self.assertEqual(count, 2)


class MultiClassBase(unittest.TestCase):
    def __init__(self, testName, method):
        super(MultiClassBase, self).__init__(testName)
        self.method = method
        x_train = np.array([[1,1,1,1,2,2],
                            [1,1,1,2,1,2],
                            [2,2,1,1,1,2],
                            [1,2,1,1,1,1],
                            [1,1,1,1,2,1],
                            [1,1,1,2,1,1],
                            [2,2,2,2,1,1],
                            [2,2,2,1,2,1],
                            [2,2,2,1,2,2],
                            [1,2,2,2,2,2],
                            [2,1,2,1,2,2],
                            [2,2,2,2,2,1],
                            [3,3,3,4,4,3],
                            [3,1,3,2,3,3],
                            [3,3,3,1,1,3],
                            [3,3,3,1,3,3],
                            [1,2,3,3,3,3],
                            [3,3,3,3,3,5]])
        y_train = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3])
        x_test = np.array([[1,1,1,1,1,1],
                           [2,2,2,2,2,2],
                           [3,3,3,3,3,3]])
        y_test = np.array([1,2,3])
        self.data = [x_train, y_train, x_test, y_test]
        self.new_y_train = np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2
                                    ,2,2,3,3,3,3,3,3,3,3,3,3,3,3])

    def test_dimensions(self):
        new_data = self.method(self.data, 6)
        correct_dims = [(36,6), (36,), (3,6), (3,)]
        new_dims = [x.shape for x in new_data]
        count = 0
        for x in range(len(correct_dims)):
            if correct_dims[x] == new_dims[x]:
                count += 1
        self.assertEqual(count, 4)

    def test_train_labels(self):
        new_data = self.method(self.data, 6)
        count = 0
        for x in new_data[1].tolist():
            if x in self.new_y_train.tolist():
                count += 1
        for x in self.new_y_train.tolist():
            if x in new_data[1].tolist():
                count += 1
        self.assertEqual(count, 72)

    def test_test_values_and_labels(self):
        new_data = self.method(self.data, 6)
        count = 0
        for x in [-1,-2]:
            if (new_data[x] == self.data[x]).all():
                count += 1
        self.assertEqual(count, 2)

def binary_suite(method):
    suite = unittest.TestSuite()
    suite.addTest(BinaryBase('test_dimensions', method))
    suite.addTest(BinaryBase('test_train_labels', method))
    suite.addTest(BinaryBase('test_test_values_and_labels', method))
    return suite

def multiclass_suite(method):
    suite = unittest.TestSuite()
    suite.addTest(MultiClassBase('test_dimensions', method))
    suite.addTest(MultiClassBase('test_train_labels', method))
    suite.addTest(MultiClassBase('test_test_values_and_labels', method))
    return suite

def get_all_tests():
    methods = [augment_data_naive, augment_data_smote, augment_data_adasyn,
               augment_data_noise]
    all_tests = unittest.TestSuite()
    for method in methods:
        all_tests.addTest(multiclass_suite(method))
        all_tests.addTest(binary_suite(method))
    return all_tests

if __name__ == "__main__":
    all_tests = get_all_tests()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
