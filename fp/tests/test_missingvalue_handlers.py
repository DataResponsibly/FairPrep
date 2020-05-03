import unittest
from pandas import read_csv

from fp.missingvalue_handlers import CompleteCaseAnalysis, ModeImputer, DataWigSimpleImputer

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.columns = ['native-country', 'occupation', 'workclass']
        self.label_name = 'income-per-year'
        self.out_path = './datawig-models/'
        self.data = read_csv('fp/tests/resource/input/data_missing.csv')
        self.count_cols = 4
        self.count_rows = 32561
        self.count_complete_rows = 30162
        
    def test_CompleteCaseAnalysis(self):
        self.handler = CompleteCaseAnalysis()
        self.handler.fit(self.data)
        result = self.handler.handle_missing(self.data)
        self.assertEqual(self.handler.name(), 'complete_case')
        self.assertEqual(type(result), type(self.data))
        self.assertEqual(result.shape, (self.count_complete_rows, self.count_cols))
        
    def test_ModeImputer(self):
        self.handler = ModeImputer(self.columns)
        self.assertEqual(self.handler.columns, self.columns)
        self.handler.fit(self.data)
        result = self.handler.handle_missing(self.data)
        self.assertEqual(self.handler.name(), 'mode_imputation')
        self.assertEqual(type(result), type(self.data))
        self.assertEqual(result.shape, (self.count_rows, self.count_cols))
        
    def test_DataWigSimpleImputer(self):
        self.handler = DataWigSimpleImputer(self.columns, self.label_name, self.out_path)
        self.assertEqual(self.handler.name(), 'datawig_imputation')
        self.assertEqual(self.handler.columns_to_impute, self.columns)
        self.assertEqual(self.handler.label_column, self.label_name)
        self.assertEqual(self.handler.out, self.out_path)
        # self.handler.fit(self.data)
        # result = self.handler.handle_missing(self.data)

if __name__ == '__main__':
    unittest.main()
