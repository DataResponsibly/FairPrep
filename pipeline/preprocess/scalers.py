"""
    Classes to scale data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pipeline.step import Step

class SK_StandardScaler(Step):

    def __init__(self, num_atts, copy=True, with_mean=True, with_std=True):
        """
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param copy: same parameter with sklearn StandardScaler
        :param with_mean: same parameter with sklearn StandardScaler
        :param with_std: same parameter with sklearn StandardScaler
        """
        self.focus_atts = num_atts
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        for ai in self.focus_atts:
            fitted_step[ai] = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std).fit(np.array(df[ai]).reshape(-1, 1))

        self.fitted_step = fitted_step

        return self

    def apply(self, df):
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "StandardScaler"

    def abbr_name(self):
        return "SS"

    def step_name(self):
        return "Scaler"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True


class SK_MinMaxScaler(Step):
    def __init__(self, num_atts, feature_range=(0, 1), copy=True):
        """
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param feature_range: same parameter with sklearn MinMaxScaler
        :param copy: same parameter with sklearn MinMaxScaler
        """
        self.focus_atts = num_atts
        self.feature_range = feature_range
        self.copy = copy

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        for ai in self.focus_atts:
            fitted_step[ai] = MinMaxScaler(feature_range=self.feature_range, copy=self.copy).fit(np.array(df[ai]).reshape(-1, 1))

        self.fitted_step = fitted_step

        return self

    def apply(self, df):
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "MinMaxScaler"

    def abbr_name(self):
        return "MS"

    def step_name(self):
        return "Scaler"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_AIF.csv")
    # cur_o = SK_StandardScaler(["month", "credit_amount"])
    cur_o = SK_MinMaxScaler(["month", "credit_amount"])

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_AIF_"+cur_o.name()+".csv", index=False)
