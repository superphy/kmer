import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import shutil
import random
import numpy as np
import pandas as pd
import string
import tempfile
import constants
import json
from utils import same_shuffle, shuffle, parse_metadata, setup_files
from utils import check_fasta, valid_file, flatten, make3D
from utils import sensitivity_specificity, parse_json, make_unique


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
        self.dir = tempfile.mkdtemp()
        self.file = self.dir + '/metadata.temp'
        self.length = 100
        self.classes = 4
        self.classifications = list(string.ascii_lowercase)
        self.datasets = ['Train', 'Test']
        data = {'Fasta':[x for x in range(self.length)],
                'Class':[self.classifications[x%self.classes] for x in range(self.length)],
                'Dataset':[self.datasets[x%len(self.datasets)] for x in range(self.length)],
                'Extra':[x for x in range(self.length)],
                'Ignore':[random.uniform(0,1) for _ in range(self.length)]}
        self.data = pd.DataFrame(data=data)
        self.data.to_csv(self.file)

    def tearDown(self):
        shutil.rmtree(self.dir)

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

    def test_default_train(self):
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

    def test_default_test(self):
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

    def test_removed_x_train(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct = [str(x) for x in range(self.length) if x%2 == 0]
        correct = [x for x in correct if int(x)%self.classes != 0]
        val = False
        if np.array_equal(np.unique(x_train, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val, msg = '\n' + str(correct) + '\n' + str(x_train))

    def test_removed_y_train(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        correct = [x for x in correct if x != 'a']
        if np.array_equal(np.unique(y_train, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_removed_x_test(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct = [str(x) for x in range(self.length) if x%2 == 1]
        correct = [x for x in correct if int(x)%self.classes != 0]
        val = False
        if np.array_equal(np.unique(x_test, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_removed_y_test(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        correct = [x for x in correct if x != 'a']
        val = False
        if np.array_equal(np.unique(y_test, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_removed_train(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct_x = [str(x) for x in range(self.length) if x%2 == 0]
        correct_x = [x for x in correct_x if int(x)%self.classes != 0]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        correct_y = [x for x in correct_y if x != 'a']
        count1 = 0
        count2 = 0
        for elem in x_train:
            index = correct_x.index(elem)
            if y_train[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

    def test_removed_test(self):
        if self.classes <= 2:
            return unittest.skip('Too few classes')
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,remove='a')
        correct_x = [str(x) for x in range(self.length) if x%2 == 1]
        correct_x = [x for x in correct_x if int(x)%self.classes != 0]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        correct_y = [x for x in correct_y if x != 'a']
        count1 = 0
        count2 = 0
        for elem in x_test:
            index = correct_x.index(elem)
            if y_test[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

    def test_ova_x_train(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct = [str(x) for x in range(self.length) if x%2 == 0]
        val = False
        if np.array_equal(np.unique(x_train, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val, msg = '\n' + str(correct) + '\n' + str(x_train))

    def test_ova_y_train(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        correct = [x if x =='a' else 'Other' for x in correct]
        if np.array_equal(np.unique(y_train, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_ova_x_test(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct = [str(x) for x in range(self.length) if x%2 == 1]
        if np.array_equal(np.unique(x_test, return_counts=True), np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_ova_y_test(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        correct = [x if x == 'a' else 'Other' for x in correct]
        val = False
        if np.array_equal(np.unique(y_test, return_counts=True),np.unique(correct, return_counts=True)):
            val = True
        self.assertTrue(val)

    def test_ova_train(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct_x = [str(x) for x in range(self.length) if x%2 == 0]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==0]
        correct_y = [x if x =='a' else 'Other' for x in correct_y]
        count1 = 0
        count2 = 0
        for elem in x_train:
            index = correct_x.index(elem)
            if y_train[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

    def test_ova_test(self):
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,one_vs_all='a')
        correct_x = [str(x) for x in range(self.length) if x%2 == 1]
        correct_y = [self.classifications[x%self.classes] for x in range(self.length) if x%2==1]
        correct_y = [x if x == 'a' else 'Other' for x in correct_y]
        count1 = 0
        count2 = 0
        for elem in x_test:
            index = correct_x.index(elem)
            if y_test[count2] == correct_y[index]:
                count1 += 1
            count2 += 1
        self.assertEqual(count1, count2)

    def test_blacklist(self):
        bl = np.unique(np.random.randint(self.length, size=(int(0.1*self.length))))
        x_train, y_train, x_test, y_test = parse_metadata(metadata=self.file,blacklist=bl)
        count1 = 0
        count2 = 0
        for elem in bl:
            count2 += 1
            if elem not in x_train and elem not in x_test:
                count1 += 1
        self.assertEqual(count1, count2)


class SetUpFiles(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.num_files = 10
        self.correct=[self.directory+'/temp%d'%x for x in range(self.num_files)]
        for x in self.correct: open(x, 'w')

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_without_backslash(self):
        files = setup_files(self.directory)
        count = 0
        for x in range(self.num_files):
            if files[x] in self.correct:
                count += 1
        for x in range(self.num_files):
            if self.correct[x] in files:
                count += 1
        self.assertEqual(count, 2*self.num_files)

    def test_with_backslash(self):
        files = setup_files(self.directory + '/')
        count = 0
        for x in range(self.num_files):
            if files[x] in self.correct:
                count += 1
        for x in range(self.num_files):
            if self.correct[x] in files:
                count += 1
        self.assertEqual(count, 2*self.num_files)

class CheckFasta(unittest.TestCase):
    def setUp(self):
        self.num_files = 10
        files = [constants.ECOLI + x for x in os.listdir(constants.ECOLI)]
        self.fasta = files[:self.num_files]
        bad_files = [constants.SOURCE + x for x in os.listdir(constants.SOURCE) if os.path.isfile(x)]
        self.non_fasta = bad_files[:self.num_files]

    def test_non_fasta(self):
        val = False
        for x in self.non_fasta:
            val = val or check_fasta(x)
        self.assertFalse(val)

    def test_fasta(self):
        val = True
        for x in self.fasta:
            val = val and check_fasta(x)
        self.assertTrue(val)

class ValidFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.bad_files = ['bad1.txt', 'bad2.txt', 'bad3.txt', 'bad4.txt']
        self.badA = self.dir + '/TEMPbad1.txt'
        self.badB = self.dir + '/TEMPbad2.txt'
        with open(self.badA, 'w') as f:
            f.write('%s\n%s'%(self.bad_files[0], self.bad_files[1]))
        with open(self.badB, 'w') as f:
            f.write('%s\n%s'%(self.bad_files[2], self.bad_files[3]))
        self.test_files = ['good3.txt', 'bad2.txt', 'bad1.txt', 'bad3.txt',
                           'good2.txt', 'good4.txt', 'bad4.txt', 'good1.txt']
        self.new_files = valid_file(self.test_files, self.badB, self.badA)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_dimensions(self):
        length = len(self.new_files)
        self.assertEqual(length, 4, msg=self.test_files)

    def test_misiing(self):
        count = 0
        for x in self.bad_files:
            if x in self.new_files:
                count += 1
        self.assertEqual(count, 0, msg=self.new_files)

class FlattenAnd3D(unittest.TestCase):
    def setUp(self):
        self.input = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
        self.output = np.array([[1,2,3],[4,5,6],[7,8,9]])
        self.flat = flatten(self.input)
        self.threeD = make3D(self.output)

    def test_flatten(self):
        val = np.array_equal(self.output, self.flat)
        self.assertTrue(val)

    def test_make3D(self):
        val = np.array_equal(self.input, self.threeD)
        self.assertTrue(val)

class SensitivitySpecificity(unittest.TestCase):
    def setUp(self):
        predicted_values = np.array([1,1,1,2,1,1,2,2,2,3,2,1,2,1])
        true_values =      np.array([1,1,2,1,1,1,2,2,2,3,3,3,2,1])
        self.results = sensitivity_specificity(predicted_values, true_values)
        self.correct = {1:{'sensitivity': 6.0/7.0,
                           'specificity': 8.0/10.0},
                        2:{'sensitivity': 5.0/6.0,
                           'specificity': 9.0/11.0},
                        3:{'sensitivity': 3.0/5.0,
                           'specificity': 11.0/11.0}}

    def test_sensitivity(self):
        count1 = 0
        count2 = 0
        for x in [1,2,3]:
            count2 += 1
            if self.results[x]['sensitivity'] == self.correct[x]['sensitivity']:
                count1 += 1
        self.assertEqual(count1, count2)

    def test_specificity(self):
        count1 = 0
        count2 = 0
        for x in [1,2,3]:
            count2 += 1
            if self.results[x]['specificity'] == self.correct[x]['specificity']:
                count1 += 1
        self.assertEqual(count1,count2)

class ParseJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp() + '/'
        self.jsonA = self.dir + 'TEMPA.json'
        self.jsonB = self.dir + 'TEMPB.json'
        self.jsonC = self.dir + 'TEMPC.json'
        with open(self.jsonA, 'w') as f:
            f.write(json.dumps([{"garbage": 123, "name": 'A1', 'garbage2': 456},
                                {"garbage": 321, "name": 'A2', 'garbage2': 654},
                                {'garbage': -1, "name": 'A3', 'garbage2': 45}]))
        with open(self.jsonB, 'w') as f:
            f.write(json.dumps([{"garbage": 123, "name": 'B1', 'garbage2': 456},
                                {"garbage": 321, "name": 'B2', 'garbage2': 654},
                                {'garbage': -1, "name": 'B3', 'garbage2': 45}]))
        with open(self.jsonC, 'w') as f:
            f.write(json.dumps([{"garbage": 123, "name": 'C1', 'garbage2': 456},
                                {"garbage": 321, "name": 'C2', 'garbage2': 654},
                                {'garbage': -1, "name": 'C3', 'garbage2': 45}]))
        with open(self.dir + 'A1.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'A2.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'A3.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'B1.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'B2.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'B3.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'C1.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'C2.temp', 'w') as f:
            f.write('>')
        with open(self.dir + 'C3.temp', 'w') as f:
            f.write('>')
        self.output = parse_json((self.jsonA, self.jsonB, self.jsonC),
                                 path=self.dir, suffix='.temp', key='name')

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_class_num(self):
        self.assertEqual(len(self.output), 3)

    def test_classA(self):
        count = 0
        values = self.output[0]
        if values[0] == self.dir + 'A1.temp':
            count += 1
        if values[1] == self.dir + 'A2.temp':
            count += 1
        if values[2] == self.dir + 'A3.temp':
            count += 1
        self.assertEqual(count, 3)

    def test_classB(self):
        count = 0
        values = self.output[1]
        if values[0] == self.dir + 'B1.temp':
            count += 1
        if values[1] == self.dir + 'B2.temp':
            count += 1
        if values[2] == self.dir + 'B3.temp':
            count += 1
        self.assertEqual(count, 3)

    def test_classC(self):
        count = 0
        values = self.output[2]
        if values[0] == self.dir + 'C1.temp':
            count += 1
        if values[1] == self.dir + 'C2.temp':
            count += 1
        if values[2] == self.dir + 'C3.temp':
            count += 1
        self.assertEqual(count, 3)


class EmptyDictionaries(unittest.TestCase):
    def setUp(self):
        self.default = parse_metadata()
        self.empty = parse_metadata(**{})

    def test_x_train(self):
        count1 = 0
        count2 = 0
        for elem in self.default[0]:
            count2+=1
            if elem in self.empty[0]:
                count1+=1
        for elem in self.empty[0]:
            count2+=1
            if elem in self.default[0]:
                count1+=1
        self.assertEqual(count1, count2)

    def test_y_train(self):
        count1 = 0
        count2 = 0
        for elem in self.default[1]:
            count2+=1
            if elem in self.empty[1]:
                count1+=1
        for elem in self.empty[1]:
            count2+=1
            if elem in self.default[1]:
                count1+=1
        self.assertEqual(count1, count2)

    def test_x_test(self):
        count1 = 0
        count2 = 0
        for elem in self.default[2]:
            count2+=1
            if elem in self.empty[2]:
                count1+=1
        for elem in self.empty[2]:
            count2+=1
            if elem in self.default[2]:
                count1+=1
        self.assertEqual(count1, count2)

    def test_y_test(self):
        count1 = 0
        count2 = 0
        for elem in self.default[3]:
            count2+=1
            if elem in self.empty[3]:
                count1+=1
        for elem in self.empty[3]:
            count2+=1
            if elem in self.default[3]:
                count1+=1
        self.assertEqual(count1, count2)


class MakeUnique(unittest.TestCase):
    def setUp(self):
        self.simple = [[1,2,3,4,5],[2,3,4,5,6]]
        self.simple_answer = [[2,3,4,5],[2,3,4,5]]
        self.complex = [[1,2,3,4,5],[5,4,3,2,1],[6,7,2,5,4],[10,11,2,7,5],[1,2,5,4,7]]
        self.complex_answer = [[2,5],[5,2],[2,5],[2,5],[2,5]]

    def test_simple(self):
        make_unique(self.simple)
        val = False
        if np.array_equal(self.simple, self.simple_answer):
            val = True
        self.assertTrue(val)

    def test_complex(self):
        make_unique(self.complex)
        val = False
        if np.array_equal(self.complex, self.complex_answer):
            val = True
        self.assertTrue(val)

if __name__=="__main__":
    loader = unittest.TestLoader()
    all_tests = loader.discover('.', pattern='test_utils.py')
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
