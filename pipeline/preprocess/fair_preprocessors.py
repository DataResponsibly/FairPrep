"""
    Class of fairness preprocessing interventions
"""

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing as Reweighing
from aif360.algorithms.preprocessing import LFR as LFR
from aif360.algorithms.preprocessing import DisparateImpactRemover
from pipeline.preprocess.preprocessor import Preprocessor

class AIF_Reweighing(Preprocessor):

    def __init__(self, df, target_col, sensitive_att):
        """
        :param df: pandas dataframe, stores the data to fit the scaler.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_value: str, the value of above target variable that represents positive outcome. default is 1.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.

        """

        cur_step = Reweighing([{sensitive_att: 0}], [{sensitive_att: 1}])
        super().__init__("@".join(["AIF_Reweighing", sensitive_att]), df, step=cur_step, fit_flag=True, weight_flag=True, sensitive_att=sensitive_att, target_col=target_col, fair_aware=True)


class AIF_DIRemover(Preprocessor):

    def __init__(self, df, target_col, sensitive_att, repair_level):
        """
        :param df: pandas dataframe, stores the data to fit the scaler.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_value: str, the value of above target variable that represents positive outcome. default is 1.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.

        """
        if repair_level is None or not isinstance(repair_level, float):
            print("Input repair_level is not valid! Should be float within [0,1]!")
            raise ValueError
        else:
            if repair_level < 0 or repair_level > 1:
                print("Input repair_level is not valid! Should be float within [0,1]!")
                raise ValueError
        self.repair_level = repair_level
        cur_step = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=sensitive_att)

        super().__init__("@".join(["AIF_DIRemover", sensitive_att]), df, step=cur_step, fit_flag=False, sensitive_att=sensitive_att, target_col=target_col, fair_aware=True)


class AIF_LFR(Preprocessor):

    def __init__(self, df, target_col, sensitive_att):
        """ NOTE: very sensitive to input data, refer the example in AIF 360 for this preprocessor
        :param df: pandas dataframe, stores the data to fit the scaler.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_value: str, the value of above target variable that represents positive outcome. default is 1.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.

        """
        # TODO: fix the bug of LFR for not returning categorical atts
        # TODO: experiment with the same data used by AIF360 tutorial to compare whether the categorical atts are returned
        cur_step = LFR([{sensitive_att: 0}], [{sensitive_att: 1}])
        super().__init__("@".join(["AIF_LFR", sensitive_att]), df, step=cur_step, fit_flag=True, sensitive_att=sensitive_att, target_col=target_col, fair_aware=True)



if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_reweigh.csv")
    # cur_o = AIF_Reweighing(data, "income-per-year", "sex")
    # cur_o = AIF_LFR(data, "income-per-year", "sex")
    cur_o = AIF_DIRemover(data, "income-per-year", "sex", 0.8)

    after_data = cur_o.apply(data)
    # for Reweighing
    # after_data, new_weights = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())
    # for Reweighing
    # print(len(new_weights))
    # print(new_weights)