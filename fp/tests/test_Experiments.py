import unittest
import pickle
from pandas import read_csv

from fp.traindata_samplers import CompleteData
from fp.missingvalue_handlers import CompleteCaseAnalysis
from fp.scalers import NoScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree
from fp.pre_processors import NoPreProcessing
from fp.post_processors import NoPostProcessing
from fp.experiments import BinaryClassificationExperiment

class testSuiteExperiments(unittest.TestCase):
    
    @unittest.mock.patch.object(BinaryClassificationExperiment, 'generate_timestamp', unittest.mock.MagicMock(return_value='2020-01-01_00-00-00-000'))
    def setUp(self):
        # User defined arguments
        fixed_random_seed = 0xbeef
        train_data_sampler = CompleteData()
        missing_value_handler = CompleteCaseAnalysis()
        numeric_attribute_scaler = NoScaler()
        learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()]
        pre_processors = [NoPreProcessing()]
        post_processors = [NoPostProcessing()]
        
        # Fixed arguments for dataset
        test_set_ratio = 0.2
        validation_set_ratio = 0.1
        label_name = 'credit'
        positive_label = 1
        numeric_attribute_names = ['month', 'credit_amount', 'residence_since', 'age', 'number_of_credits',
                                   'people_liable_for']
        categorical_attribute_names = ['credit_history', 'savings', 'employment']
        attributes_to_drop_names = ['personal_status', 'status', 'purpose', 'investment_as_income_percentage',
                                    'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level',
                                    'telephone', 'foreign_worker']
        protected_attribute_names = ['sex']
        privileged_classes = [[1.0]]
        privileged_groups = [{'sex': 1.0}]
        unprivileged_groups = [{'sex': 0.0}]
        dataset_metadata = {'label_maps': [{1.0: 1, 0.0: 0}], 
                            'protected_attribute_maps': [{1.0: 'male', 0.0: 'female'}]
                            }
        dataset_name = 'test_dataset'
        
        # Constructor call
        self.experiment = BinaryClassificationExperiment(fixed_random_seed, test_set_ratio, validation_set_ratio,
                                                         label_name, positive_label, numeric_attribute_names,
                                                         categorical_attribute_names, attributes_to_drop_names,
                                                         train_data_sampler, missing_value_handler, numeric_attribute_scaler,
                                                         learners, pre_processors, post_processors,
                                                         protected_attribute_names, privileged_classes, privileged_groups,
                                                         unprivileged_groups, dataset_metadata, dataset_name)
        self.data = read_csv('fp/tests/resource/input/data.csv')
        self.annotated_train_data = pickle.load(open('fp/tests/resource/input/data_annotated.obj', 'rb'))
    
    def test_constructor(self):
        self.assertEqual(self.experiment.fixed_random_seed, 0xbeef)
        self.assertEqual(self.experiment.test_set_ratio, 0.2)
        self.assertEqual(self.experiment.validation_set_ratio, 0.1)
        self.assertEqual(self.experiment.label_name, 'credit')
        self.assertEqual(self.experiment.positive_label, 1)
        self.assertEqual(self.experiment.numeric_attribute_names, ['month', 'credit_amount', 'residence_since',
                                                                   'age', 'number_of_credits', 'people_liable_for'])
        self.assertEqual(self.experiment.categorical_attribute_names, ['credit_history', 'savings', 'employment'])
        self.assertEqual(self.experiment.attributes_to_drop_names, ['personal_status', 'status', 'purpose', 
                                                                    'investment_as_income_percentage', 'other_debtors',
                                                                    'property', 'installment_plans', 'housing',
                                                                    'skill_level', 'telephone', 'foreign_worker'])
        self.assertEqual(type(self.experiment.train_data_sampler), CompleteData)
        self.assertEqual(type(self.experiment.missing_value_handler), CompleteCaseAnalysis)
        self.assertEqual(type(self.experiment.numeric_attribute_scaler), NoScaler)
        self.assertEqual(len(self.experiment.learners), 2)
        self.assertEqual(type(self.experiment.learners[0]), NonTunedLogisticRegression)
        self.assertEqual(type(self.experiment.learners[1]), NonTunedDecisionTree)
        self.assertEqual(len(self.experiment.pre_processors), 1)
        self.assertEqual(type(self.experiment.pre_processors[0]), NoPreProcessing)
        self.assertEqual(len(self.experiment.post_processors), 1)
        self.assertEqual(type(self.experiment.post_processors[0]), NoPostProcessing)
        self.assertEqual(self.experiment.protected_attribute_names, ['sex'])
        self.assertEqual(self.experiment.privileged_classes, [[1.0]])
        self.assertEqual(self.experiment.privileged_groups, [{'sex': 1.0}])
        self.assertEqual(self.experiment.unprivileged_groups, [{'sex': 0.0}])
        self.assertEqual(self.experiment.dataset_metadata, {'label_maps': [{1.0: 1, 0.0: 0}], 
                                                            'protected_attribute_maps': [{1.0: 'male', 0.0: 'female'}]
                                                            })
        self.assertEqual(self.experiment.dataset_name, 'test_dataset')
        self.assertEqual(self.experiment.log_path, 'logs/')
        self.assertEqual(self.experiment.exec_timestamp, '2020-01-01_00-00-00-000')

    def test_unique_file_name(self):
        self.assertEqual(self.experiment.unique_file_name(self.experiment.learners[0], 
                                                         self.experiment.pre_processors[0], 
                                                         self.experiment.post_processors[0]),
            'test_dataset__LogisticRegression-notuning__complete_case__complete_data__no_scaler__no_pre_processing__no_post_processing')
        self.assertEqual(self.experiment.unique_file_name(self.experiment.learners[1], 
                                                         self.experiment.pre_processors[0], 
                                                         self.experiment.post_processors[0]),
            'test_dataset__DecisionTree-notuning__complete_case__complete_data__no_scaler__no_pre_processing__no_post_processing')

    def test_generate_file_path(self):
        self.assertEqual(self.experiment.generate_file_path(''), 'logs/2020-01-01_00-00-00-000_test_dataset/')
        self.assertEqual(self.experiment.generate_file_path('test.csv'), 'logs/2020-01-01_00-00-00-000_test_dataset/test.csv')

if __name__ == '__main__':
    unittest.main()
    
