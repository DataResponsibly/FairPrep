import unittest

from fp.traindata_samplers import CompleteData
from fp.missingvalue_handlers import CompleteCaseAnalysis
from fp.scalers import NamedStandardScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree
from fp.pre_processors import NoPreProcessing
from fp.post_processors import NoPostProcessing
from fp.dataset_experiments import AdultDatasetWhiteMaleExperiment, AdultDatasetMaleExperiment, AdultDatasetWhiteExperiment
from fp.dataset_experiments import PropublicaDatasetWhiteExperiment, GermanCreditDatasetSexExperiment, RicciRaceExperiment
from fp.dataset_experiments import GiveMeSomeCreditExperiment

class TestSuiteDatasets(unittest.TestCase):
    
    def test_AdultDatasetWhiteMaleExperiment(self):
        self.experiment = AdultDatasetWhiteMaleExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

    def test_AdultDatasetMaleExperiment(self):
        self.experiment = AdultDatasetMaleExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

    def test_AdultDatasetWhiteExperiment(self):
        self.experiment = AdultDatasetWhiteExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

    def test_PropublicaDatasetWhiteExperiment(self):
        self.experiment = PropublicaDatasetWhiteExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

    def test_GermanCreditDatasetSexExperiment(self):
        self.experiment = GermanCreditDatasetSexExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

    def test_RicciRaceExperiment(self):
        self.experiment = RicciRaceExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()
        
    def test_GiveMeSomeCreditExperiment(self):
        self.experiment = GiveMeSomeCreditExperiment(
            fixed_random_seed = 0xabcd,
            train_data_sampler = CompleteData(),
            missing_value_handler = CompleteCaseAnalysis(),
            numeric_attribute_scaler = NamedStandardScaler(),
            learners = [NonTunedLogisticRegression(), NonTunedDecisionTree()],
            pre_processors = [NoPreProcessing()],
            post_processors = [NoPostProcessing()]
        )
        self.experiment.run()

if __name__ == '__main__':
    unittest.main()
