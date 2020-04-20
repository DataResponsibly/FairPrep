import os
import unittest

from fp.traindata_samplers import CompleteData
from fp.scalers import NoScaler
from fp.learners import NonTunedLogisticRegression, NonTunedDecisionTree
from fp.pre_processors import NoPreProcessing
from fp.post_processors import NoPostProcessing
from fp.experiments import BinaryClassificationExperiment

class testSummer(unittest.TestCase):
  # Test 01
  def testSum01(self):
    num1 = 4
    num2 = 10
    self.assertEqual(14, num1+num2)
    
if __name__ == '__main__':
    unittest.main()
    
