import unittest
import numpy as np
from kmerprediction.models import neural_network, support_vector_machine, random_forest


class NeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.x_train = np.random.rand(10, 10)
        self.y_train = np.random.randint(0, 3, size=10)
        self.x_test = np.random.randn(10, 10)
        self.y_test = np.array([], dtype='float64')
        self.y_test_validate = np.random.randint(0, 3, size=10)

    def test_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test,
                      self.y_test_validate]
        output, feature_names = neural_network(input_data)
        count = 0
        if feature_names is not None:
            count += 1

        if not isinstance(output, float):
            count += 1

        if output > 1.0 or output < 0.0:
            count += 1

        self.assertEqual(count, 0)

    def test_non_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test, self.y_test]
        fn = ['a', 'b', 'c', 'd']
        output, feature_names = neural_network(input_data, feature_names=fn,
                                               validate=False)
        count = 0
        if feature_names is not None:
            count += 1

        if not isinstance(output, (list, np.ndarray)):
            count += 1

        if len(list(output)) != 10:
            count += 1

        self.assertEqual(count, 0)


class SupportVectorMachine(unittest.TestCase):
    def setUp(self):
        self.x_train = np.random.rand(10, 10)
        self.y_train = np.random.randint(0, 3, size=10)
        self.x_test = np.random.randn(10, 10)
        self.y_test = np.array([], dtype='float64')
        self.y_test_validate = np.random.randint(0, 3, size=10)

    def test_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test,
                      self.y_test_validate]
        fn = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        output, features = support_vector_machine(input_data, feature_names=fn)
        count = 0
        if not isinstance(features, dict):
            count += 1

        if not isinstance(output, float):
            count += 1

        if count > 1.0 or count < 0.0:
            count += 1

        if sorted(features.keys()) != fn:
            count += 1

        self.assertEqual(count, 0)

    def test_non_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test, self.y_test]
        output, features = support_vector_machine(input_data, validate=False)
        count = 0
        if features is not None:
            count += 1

        if not isinstance(output, (list, np.ndarray)):
            count += 1

        if len(list(output)) != 10:
            count += 1

        self.assertEqual(count, 0)


class RandomForest(unittest.TestCase):
    def setUp(self):
        self.x_train = np.random.rand(10, 10)
        self.y_train = np.random.randint(0, 3, size=10)
        self.x_test = np.random.randn(10, 10)
        self.y_test = np.array([], dtype='float64')
        self.y_test_validate = np.random.randint(0, 3, size=10)

    def test_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test,
                      self.y_test_validate]
        fn = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        output, features = random_forest(input_data, feature_names=fn)
        count = 0
        if not isinstance(features, dict):
            count += 1

        if not isinstance(output, float):
            count += 1

        if count > 1.0 or count < 0.0:
            count += 1

        if sorted(features.keys()) != fn:
            count += 1

        self.assertEqual(count, 0)

    def test_non_validate(self):
        input_data = [self.x_train, self.y_train, self.x_test, self.y_test]
        output, features = random_forest(input_data, validate=False)
        count = 0
        if features is not None:
            count += 1

        if not isinstance(output, (list, np.ndarray)):
            count += 1

        if len(list(output)) != 10:
            count += 1

        self.assertEqual(count, 0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_models.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
