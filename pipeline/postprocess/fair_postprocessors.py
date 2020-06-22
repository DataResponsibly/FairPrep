"""
    Classes for post-process data and model outcomes
"""

import pandas as pd
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification
from pipeline.postprocess.postprocessor import Postprocessor

class AIF_EqOddsPostprocessing(Postprocessor):

    def __init__(self, df, target_col, sensitive_att, threshold=0.5, seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the postprocessor.
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param threshold: float in [0, 1], the classification threshold to generate the predicted class label.
        :param seed: integer, the seed for random state.
        """

        cur_step = EqOddsPostprocessing([{sensitive_att: 0}], [{sensitive_att: 1}], seed)
        super().__init__("@".join(["AIF_EqOddsPostprocessing", sensitive_att]), cur_step, df, sensitive_att, target_col, input_score=False, clf_threshold=threshold)


class AIF_CalibratedEqOddsPostprocessing(Postprocessor):

    def __init__(self, df, target_col, sensitive_att, threshold=0.5, seed=0, cost_constraint='weighted'):
        """
        :param df: pandas dataframe, stores the data to fit the postprocessor.
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param threshold: float in [0, 1], the classification threshold to generate the predicted class label.
        :param seed: integer, the seed for random state.
        :param cost_constraint: str, the fairness constraints format, value from [fpr, fnr, weighted].
                                The same parameter as in aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing.
        """

        cur_step = CalibratedEqOddsPostprocessing([{sensitive_att: 0}], [{sensitive_att: 1}], cost_constraint=cost_constraint, seed=seed)
        super().__init__("@".join(["AIF_CalibratedEqOddsPostprocessing", sensitive_att]), cur_step, df, sensitive_att, target_col, input_score=True, clf_threshold=threshold)

class AIF_RejectOptionPostprocessing(Postprocessor):

    def __init__(self, df, target_col, sensitive_att, threshold=0.5):
        """
        :param df: pandas dataframe, stores the data to fit the postprocessor.
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param threshold: float in [0, 1], the classification threshold to generate the predicted class label.
        """
        # TODO: fix the bug that reject option doesn't return results
        cur_step = RejectOptionClassification([{sensitive_att: 0}], [{sensitive_att: 1}])
        super().__init__("@".join(["AIF_RejectOptionClassification", sensitive_att]), cur_step, df, sensitive_att, target_col, input_score=True, clf_threshold=threshold)



if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_post.csv")
    # cur_o = AIF_RejectOptionPostprocessing(data, "income-per-year", "sex")
    cur_o = AIF_EqOddsPostprocessing(data, "income-per-year", "sex")
    # cur_o = AIF_CalibratedEqOddsPostprocessing(data, "income-per-year", "sex")

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())