import unittest
import pickle

from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.validation_dataset = pickle.load(open('fp/tests/resource/input/data_validation.obj', 'rb'))
        self.validation_dataset_with_predictions = pickle.load(open('fp/tests/resource/input/data_validation_with_predictions.obj', 'rb'))
        self.testset_with_predictions = pickle.load(open('fp/tests/resource/input/data_test_with_predictions.obj', 'rb'))
        self.seed = 0xbeef
        self.privileged_groups = [{'sex': 1.0}]
        self.unprivileged_groups = [{'sex': 0.0}]
        
    def test_NoPostProcessing(self):
        self.post_processor = NoPostProcessing()
        result = self.post_processor.post_process(self.validation_dataset, 
                                                self.validation_dataset_with_predictions, 
                                                self.testset_with_predictions,
                                                self.seed,
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.post_processor.name(), 'no_post_processing')
        self.assertEqual(result, self.testset_with_predictions)
        
    def test_RejectOptionPostProcessing(self):
        self.post_processor = RejectOptionPostProcessing()
        result = self.post_processor.post_process(self.validation_dataset, 
                                                self.validation_dataset_with_predictions, 
                                                self.testset_with_predictions,
                                                self.seed,
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.post_processor.name(), 'reject_option')
        self.assertEqual(type(result), type(self.testset_with_predictions))
        
    def test_EqualOddsPostProcessing(self):
        self.post_processor = EqualOddsPostProcessing()
        result = self.post_processor.post_process(self.validation_dataset, 
                                                self.validation_dataset_with_predictions, 
                                                self.testset_with_predictions,
                                                self.seed,
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.post_processor.name(), 'eq_odds')
        self.assertEqual(type(result), type(self.testset_with_predictions))
        
    def test_CalibratedEqualOddsPostProcessing(self):
        self.post_processor = CalibratedEqualOddsPostProcessing()
        result = self.post_processor.post_process(self.validation_dataset, 
                                                self.validation_dataset_with_predictions, 
                                                self.testset_with_predictions,
                                                self.seed,
                                                self.privileged_groups, 
                                                self.unprivileged_groups)
        self.assertEqual(self.post_processor.name(), 'calibrated_eq_odds')
        self.assertEqual(type(result), type(self.testset_with_predictions))
        
if __name__ == '__main__':
    unittest.main()
