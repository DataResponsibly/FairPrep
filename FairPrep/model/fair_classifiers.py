"""
    Classes of fair supervised binary classifiers.
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover
from FairPrep.step import Step

import warnings
warnings.filterwarnings("ignore")

class AIF_AdversarialDebiasing(Step):

    def __init__(self, target_col, sensitive_att, seed):
        """
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param seed: integer, the seed for random process.
        """

        import tensorflow as tf
        self.sess = tf.Session()

        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.fitted_step = None

    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        self.fitted_step = AdversarialDebiasing(unprivileged_groups=[{self.sensitive_att: 0}], privileged_groups=[{self.sensitive_att: 1}], scope_name='debiased_classifier', debias=True, sess=self.sess, seed=self.seed).fit(aif_df)

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "AdversarialDebiasing"

    def abbr_name(self):
        return "AD"

    def step_name(self):
        return "FairInprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_MetaFairClassifier(Step):

    def __init__(self, target_col, sensitive_att, fairness_penalty=0.8, fair_metric="sr"):
        """
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param fairness_penalty: float in [0,1], fairness penalty parameter. default is 0.8. The same parameter in aif360.algorithms.inprocessing.MetaFairClassifier.
        :param fair_metric: str, fairness metric used in this method. Value from ["fdr" (false discovery rate ratio), "sr" (statistical rate/disparate impact)].
                            The same parameter in aif360.algorithms.inprocessing.MetaFairClassifier.
        """

        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.fairness_penalty = fairness_penalty
        self.fair_metric = fair_metric

        self.fitted_step = None


    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col],
                                    protected_attribute_names=[self.sensitive_att])
        self.fitted_step = MetaFairClassifier(tau=self.fairness_penalty, sensitive_attr=self.sensitive_att, type=self.fair_metric).fit(aif_df)

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col],
                                         protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "MetaFairClassifier"

    def abbr_name(self):
        return "MFC"

    def step_name(self):
        return "FairInprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_PrejudiceRemover(Step):

    def __init__(self, target_col, sensitive_att, fairness_penalty=1.0):
        """
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param fairness_penalty: float in [0,1], fairness penalty parameter. default is 1. The same parameter in aif360.algorithms.inprocessing.PrejudiceRemover.

        """
        # TODO: fix the bug that cannot import lib of 'getoutput'
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.fairness_penalty = fairness_penalty

        self.fitted_step = None

    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col],
                                    protected_attribute_names=[self.sensitive_att])
        self.fitted_step = PrejudiceRemover(eta=self.fairness_penalty, sensitive_attr=self.sensitive_att, class_attr=self.target_col).fit(aif_df)

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col],
                                         protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "PrejudiceRemover"

    def abbr_name(self):
        return "PR"

    def step_name(self):
        return "FairInprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_pre_encoded.csv")
    # cur_o = AIF_AdversarialDebiasing("credit", "sex")
    cur_o = AIF_MetaFairClassifier("credit", "sex")
    # cur_o = AIF_PrejudiceRemover("credit", "sex") # not working TODO: fix

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_after_" + cur_o.name() + ".csv", index=False)
