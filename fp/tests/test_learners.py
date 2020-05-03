import unittest
import pickle
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from aif360.algorithms.inprocessing import AdversarialDebiasing as AdversarialDebiasing360

from fp.learners import LogisticRegression, DecisionTree, NonTunedLogisticRegression, NonTunedDecisionTree, AdversarialDebiasing

class TestSuitePreProcessors(unittest.TestCase):

    def setUp(self):
        self.annotated_data = pickle.load(open('fp/tests/resource/input/data_annotated.obj', 'rb'))
        self.seed = 0xbeef
        self.privileged_groups = [{'sex': 1.0}]
        self.unprivileged_groups = [{'sex': 0.0}]
        self.session = tf.Session()
        
    def test_LogisticRegression(self):
        self.learner = LogisticRegression()
        self.assertEqual(self.learner.name(), 'LogisticRegression')
        
    def test_DecisionTree(self):
        self.learner = DecisionTree()
        self.assertEqual(self.learner.name(), 'DecisionTree')
        
    def test_NonTunedLogisticRegression(self):
        self.learner = NonTunedLogisticRegression()
        result = self.learner.fit_model(self.annotated_data, self.seed)
        self.assertEqual(self.learner.name(), 'LogisticRegression-notuning')
        self.assertEqual(type(result), type(SGDClassifier()))
        
    def test_NonTunedDecisionTree(self):
        self.learner = NonTunedDecisionTree()
        result = self.learner.fit_model(self.annotated_data, self.seed)
        self.assertEqual(self.learner.name(), 'DecisionTree-notuning')
        self.assertEqual(type(result), type(DecisionTreeClassifier()))
        
    def test_AdversarialDebiasing(self):
        self.learner = AdversarialDebiasing(self.session, self.privileged_groups, self.unprivileged_groups)
        test_learner = AdversarialDebiasing360(privileged_groups=self.privileged_groups,
                                               unprivileged_groups=self.unprivileged_groups,
                                               scope_name='debiased_classifier',
                                               debias=True,
                                               sess=self.session,
                                               seed=self.seed)
        result = self.learner.fit_model(self.annotated_data, self.seed)
        self.assertEqual(self.learner.name(), 'AdversarialDebiasing')
        self.assertEqual(type(result), type(test_learner))
        self.assertEqual(self.learner.needs_annotated_data_for_prediction(), True)

if __name__ == '__main__':
    unittest.main()
