"""
    Classes of fair supervised binary classifiers.
"""

import pandas as pd
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover
from pipeline.model.inprocessor import Model
import warnings
warnings.filterwarnings("ignore")

class AIF_AdversarialDebiasing(Model):

    def __init__(self, df, target_col, sensitive_att, seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the fair classifier.
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param seed: integer, random seed.

        """

        import tensorflow as tf
        sess = tf.Session()
        cur_step = AdversarialDebiasing(unprivileged_groups=[{sensitive_att: 0}], privileged_groups=[{sensitive_att: 1}], scope_name='debiased_classifier', debias=True, sess=sess, seed=seed)
        super().__init__("@".join(["AIF_AdversarialDebiasing", sensitive_att]), cur_step, df, target_col, sensitive_att=sensitive_att, fair_aware=True)


class AIF_MetaFairClassifier(Model):

    def __init__(self, df, target_col, sensitive_att, fairness_penalty=0.8, fair_metric="sr"):
        """
        :param df: pandas dataframe, stores the data to fit the fair classifier.
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param fairness_penalty: float in [0,1], fairness penalty parameter. default is 0.8. The same parameter in aif360.algorithms.inprocessing.MetaFairClassifier.
        :param fair_metric: str, fairness metric used in this method. Value from ["fdr" (false discovery rate ratio), "sr" (statistical rate/disparate impact)].
                            The same parameter in aif360.algorithms.inprocessing.MetaFairClassifier.
        """

        cur_step = MetaFairClassifier(tau=fairness_penalty, sensitive_attr=sensitive_att, type=fair_metric)
        super().__init__("@".join(["AIF_MetaFairClassifier", sensitive_att]), cur_step, df, target_col, sensitive_att=sensitive_att, fair_aware=True)

class AIF_PrejudiceRemover(Model):

    def __init__(self, df, target_col, sensitive_att, fairness_penalty=1.0):
        """
        :param df: pandas dataframe, stores the data to fit the fair classifier.
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param fairness_penalty: float in [0,1], fairness penalty parameter. default is 1. The same parameter in aif360.algorithms.inprocessing.PrejudiceRemover.

        """
        # TODO: fix the bug that cannot import lib of 'getoutput'
        cur_step = PrejudiceRemover(eta=fairness_penalty, sensitive_attr=sensitive_att, class_attr=target_col)
        super().__init__("@".join(["AIF_PrejudiceRemover", sensitive_att]), cur_step, df, target_col, sensitive_att=sensitive_att, fair_aware=True)


if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_reweigh.csv")
    cur_o = AIF_AdversarialDebiasing(data, "income-per-year", "sex")
    # cur_o = AIF_MetaFairClassifier(data, "income-per-year", "sex")
    # cur_o = AIF_PrejudiceRemover(data, "income-per-year", "sex")

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())