"""
    Classes for post-process data and model outcomes
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification
from FairPrep.step import Step

class NoFairPostprocessor(Step):
    def __init__(self):
        self.fitted_step = None
    def fit(self, df):
        pass

    def apply(self, df): # to have consist format of the output data
        return df

    def name(self):
        return "NoPostprocessor"

    def abbr_name(self):
        return "NP"

    def step_name(self):
        return "FairPostprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return True

    def fit_only_on_train(self):
        return False

class AIF_EqOddsPostprocessing(Step):

    def __init__(self, target_col, sensitive_att, seed, threshold=0.5):
        """
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param seed: integer, the seed for random process.
        :param threshold: float in [0, 1], the classification threshold to generate the predicted class label.
        """
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.clf_threshold = threshold
        self.fitted_step = None


    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_true_df = BinaryLabelDataset(df=df.drop(columns=[self.pred_target_col]), label_names=[self.target_col],
                                         protected_attribute_names=[self.sensitive_att])

        aif_pred_df = aif_true_df.copy()
        aif_pred_df.labels = np.array([int(x >= self.clf_threshold) for x in df[self.pred_target_col]])

        self.fitted_step = EqOddsPostprocessing([{self.sensitive_att: 0}], [{self.sensitive_att: 1}], self.seed).fit(aif_true_df, aif_pred_df)

        return self

    def apply(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        df["pred_label_" + self.target_col] = [int(x >= self.clf_threshold) for x in df[self.pred_target_col]]
        aif_pred_df = BinaryLabelDataset(df=df.drop(columns=[self.pred_target_col]),
                                             label_names=["pred_label_" + self.target_col],
                                             protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "Eq_odds"

    def abbr_name(self):
        return "EQ"

    def step_name(self):
        return "FairPostprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_CalibratedEqOddsPostprocessing(Step):

    def __init__(self, target_col, sensitive_att, seed, cost_constraint='weighted'):
        """
        :param df: pandas dataframe, stores the data to fit the postprocessor.
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        :param seed: integer, the seed for random process.
        :param cost_constraint: str, the fairness constraints format, value from [fpr, fnr, weighted].
                                The same parameter as in aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing.
        """
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.cost_constraint = cost_constraint
        self.fitted_step = None

    def fit(self, df):

        aif_true_df = BinaryLabelDataset(df=df.drop(columns=[self.pred_target_col]), label_names=[self.target_col],
                                         protected_attribute_names=[self.sensitive_att])
        aif_pred_df = aif_true_df.copy()
        aif_pred_df.scores = df[self.pred_target_col]
        self.fitted_step = CalibratedEqOddsPostprocessing([{self.sensitive_att: 0}], [{self.sensitive_att: 1}], cost_constraint=self.cost_constraint, seed=self.seed).fit(aif_true_df, aif_pred_df)
        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], scores_names=[self.pred_target_col],
                                             protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)
        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "Calibrated_eq_odds"

    def abbr_name(self):
        return "CEQ"

    def step_name(self):
        return "FairPostprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_RejectOptionPostprocessing(Step):

    def __init__(self, target_col, sensitive_att):
        """
        :param df: pandas dataframe, stores the data to fit the postprocessor.
        :param target_col: str, the name of the target variable in above data. Assume 1 represents the favorable class.
        :param sensitive_att: str, the name of a sensitive attribute in above data. If none, call auto_detection to update. Value 0 represent protected.
        """
        # TODO: fix the bug that reject option outputs error

        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name

        self.fitted_step = None

    def fit(self, df):
        aif_true_df = BinaryLabelDataset(df=df.drop(columns=[self.pred_target_col]), label_names=[self.target_col],
                                         protected_attribute_names=[self.sensitive_att])
        aif_pred_df = aif_true_df.copy()
        aif_pred_df.scores = df[self.pred_target_col]
        self.fitted_step = RejectOptionClassification([{self.sensitive_att: 0}], [{self.sensitive_att: 1}]).fit(aif_true_df, aif_pred_df)
        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], scores_names=[self.pred_target_col],
                                         protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.fitted_step.predict(aif_pred_df)
        after_df, _ = after_aif_df.convert_to_dataframe()
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df

    def name(self):
        return "Reject_option"

    def abbr_name(self):
        return "RO"

    def step_name(self):
        return "FairPostprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_pre_encoded_after_model.csv")
    # cur_o = AIF_RejectOptionPostprocessing("credit", "sex")
    # cur_o = AIF_EqOddsPostprocessing("credit", "sex", 0)
    cur_o = AIF_CalibratedEqOddsPostprocessing("credit", "sex", 0)


    cur_o.fit(data)
    after_data = cur_o.apply(data)

    after_data.to_csv("../../data/german_after_" + cur_o.name() + ".csv", index=False)
