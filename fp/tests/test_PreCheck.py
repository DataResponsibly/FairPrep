import unittest
import time
from datetime import datetime

from fp.traindata_samplers import CompleteData, BalancedExamplesSampler
from fp.missingvalue_handlers import CompleteCaseAnalysis, ModeImputer, DataWigSimpleImputer
from fp.scalers import NoScaler, NamedStandardScaler, NamedMinMaxScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree, DecisionTree, LogisticRegression, AdversarialDebiasing
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing
from fp.pre_processors import NoPreProcessing, Reweighing, DIRemover, LFR
from fp.experiments import BinaryClassificationExperiment

@unittest.mock.patch('time.time', unittest.mock.MagicMock(return_value=datetime(2020, 1, 1, 0, 0, 0, 000000).timestamp()))
class test_pre_check(unittest.TestCase):
    def test_unittest(self):
        num1 = 3
        num2 = 10
        self.assertEqual(num1+num2, 13)
        
    def test_mock_timestamp(self):
        self.assertEqual(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3], '2020-01-01_00-00-00-000')
        
if __name__ == '__main__':
    unittest.main()
