import unittest
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fp.scalers import NoScaler, NamedStandardScaler, NamedMinMaxScaler

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.numeric_attribute_names = ['credit_amount', 'age']
        self.data = read_csv('fp/tests/resource/input/data.csv')
        
    def test_NoScaler(self):
        self.scaler = NoScaler()
        for numeric_attribute in self.numeric_attribute_names:
            numeric_attribute_data = np.array(self.data[numeric_attribute]).reshape(-1, 1)
            self.scaler = NoScaler()
            self.scaler.fit(numeric_attribute_data)
            result = self.scaler.transform(numeric_attribute_data)
            self.assertEqual(self.scaler.name(), 'no_scaler')
            self.assertEqual(np.array_equal(result, numeric_attribute_data), True)
            
    def test_NamedStandardScaler(self):
        self.scaler = NamedStandardScaler()
        self.assertEqual(self.scaler.name(), 'standard_scaler')
        self.assertEqual(self.scaler.with_mean, True)
        self.assertEqual(self.scaler.with_std, True)
        self.assertEqual(NamedStandardScaler.fit, StandardScaler.fit)
        self.assertEqual(NamedStandardScaler.transform, StandardScaler.transform)
        
    def test_NamedMinMaxScaler(self):
        self.scaler = NamedMinMaxScaler()
        self.assertEqual(self.scaler.name(), 'minmax_scaler')
        self.assertEqual(self.scaler.feature_range, (0, 1))
        self.assertEqual(self.scaler.copy, True)
        self.assertEqual(NamedMinMaxScaler.fit, MinMaxScaler.fit)
        self.assertEqual(NamedMinMaxScaler.transform, MinMaxScaler.transform)
        
if __name__ == '__main__':
    unittest.main()
