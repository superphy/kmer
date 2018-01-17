import unittest
import numpy as np
from collections import Counter
from data_augmentation import augment_data_naive,augment_data_smote
from data_augmentation import augment_data_adasyn,augment_data_noise


class AugmentData(unittest.TestCase):
    def setUp(self):
        self.method = augment_data_naive
        self.classes = 3
        self.original_samples = 64
        self.features = 6
        x_train = np.random.randint(10, size=(self.original_samples,self.features))
        y_train = np.random.randint(self.classes, size=self.original_samples)
        x_test = np.random.randint(4, size=(4,self.features))
        y_test = np.random.randint(self.classes, size=(4))
        self.data = [x_train, y_train, x_test, y_test]
        (values,counts) = np.unique(y_train, return_counts=True)
        self.new_samples = 2*max(counts)
        self.new_data = self.method(self.data, self.new_samples)
        self.correct_y_train = []
        for x in range(len(values)):
            self.correct_y_train.extend([values[x]]*(counts[x]+self.new_samples))

    def test_dimensions(self):
        tot_samples = self.original_samples + (self.new_samples*self.classes)
        correct_dims=[(tot_samples,self.data[0].shape[1]),(tot_samples,),self.data[2].shape,self.data[3].shape]
        new_dims = [x.shape for x in self.new_data]
        val = np.array_equal(correct_dims, new_dims)
        self.assertTrue(val)

    def test_train_labels(self):
        correct_counts = np.bincount(self.correct_y_train)
        new_counts = np.bincount(self.new_data[1])
        val = np.array_equal(correct_counts, new_counts)
        self.assertTrue(val)

    def test_test_values_and_labels(self):
        count1 = 0
        count2 = 0
        for x in [-1,-2]:
            count2 += 1
            if np.array_equal(self.new_data[x], self.data[x]):
                count1 += 1
        self.assertEqual(count1, count2)

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_data_augmentation.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
