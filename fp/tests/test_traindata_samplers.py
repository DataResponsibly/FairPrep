import unittest
from pandas import read_csv

from fp.traindata_samplers import CompleteData, BalancedExamplesSampler

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.label_name = 'credit'
        self.positive_label = 1
        self.n = 600
        self.random_state = 0xbeef
        self.data = read_csv('fp/tests/resource/input/data.csv')
        
    def test_CompleteData(self):
        self.data_sampler = CompleteData()
        result = self.data_sampler.sample(self.data)
        self.assertEqual(self.data_sampler.name(), 'complete_data')
        self.assertEqual(result.equals(self.data), True)
        
    def test_BalancedExamplesSampler(self):
        self.data_sampler = BalancedExamplesSampler(self.label_name, 
                                                    self.positive_label, 
                                                    self.n, 
                                                    self.random_state)
        result = self.data_sampler.sample(self.data)
        self.assertEqual(self.data_sampler.name(), 'balanced_data_{}'.format(self.n))
        self.assertEqual(self.data_sampler.label_name, self.label_name)
        self.assertEqual(self.data_sampler.positive_label, self.positive_label)
        self.assertEqual(self.data_sampler.n, self.n)
        self.assertEqual(self.data_sampler.random_state, self.random_state)
        self.assertEqual(result.shape, (self.n, 22))
        self.assertEqual(result[result[self.label_name]==self.positive_label].shape[0], 300)
        self.assertEqual(result[result[self.label_name]!=self.positive_label].shape[0], 300)
        self.assertEqual(type(result), type(self.data))
        
if __name__ == '__main__':
    unittest.main()
