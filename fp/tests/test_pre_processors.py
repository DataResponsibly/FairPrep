import unittest
import pickle

from fp.pre_processors import NoPreProcessing, Reweighing, DIRemover, LFR

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.privileged_groups = [{'sex': 1.0}]
        self.unprivileged_groups = [{'sex': 0.0}]
        self.repair_level = 1.0
        self.annotated_data = pickle.load(open('fp/tests/resource/input/data_annotated.obj', 'rb'))
        
    def test_NoPreProcessing(self):
        self.pre_processor = NoPreProcessing()
        result = self.pre_processor.pre_process(self.annotated_data, 
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.pre_processor.name(), 'no_pre_processing')
        self.assertEqual(result, self.annotated_data)
        
    def test_Reweighing(self):
        self.pre_processor = Reweighing()
        result = self.pre_processor.pre_process(self.annotated_data, 
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.pre_processor.name(), 'reweighing')
        self.assertNotEqual(result, self.annotated_data)
        self.assertEqual(type(result), type(self.annotated_data))
        self.assertEqual(result.convert_to_dataframe()[0].shape, 
                         self.annotated_data.convert_to_dataframe()[0].shape)
        
    def test_DIRemover(self):
        self.pre_processor = DIRemover(self.repair_level)
        result = self.pre_processor.pre_process(self.annotated_data, 
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.pre_processor.name(),
                         'diremover-' + str(self.repair_level))
        self.assertNotEqual(result, self.annotated_data)
        self.assertEqual(type(result), type(self.annotated_data))
        self.assertEqual(result.convert_to_dataframe()[0].shape, 
                         self.annotated_data.convert_to_dataframe()[0].shape)
        
    def test_LFR(self):
        self.pre_processor = LFR()
        result = self.pre_processor.pre_process(self.annotated_data, 
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.pre_processor.name(), 'LFR')
        self.assertNotEqual(result, self.annotated_data)
        self.assertEqual(type(result), type(self.annotated_data))
        self.assertEqual(result.convert_to_dataframe()[0].shape, 
                         self.annotated_data.convert_to_dataframe()[0].shape)
        
if __name__ == '__main__':
    unittest.main()
