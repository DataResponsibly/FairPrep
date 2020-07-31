"""
    Classes to discretize numerical attributes into categorical attributes.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from pipeline.step import Step

class NoBinarizer(Step):
    def __init__(self):
        self.fitted_step = None

    def fit(self, df):
        pass

    def apply(self, df):
        return df

    def name(self):
        return "NoBinarizer"

    def abbr_name(self):
        return "NB"

    def step_name(self):
        return "Categorizer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

class SK_Discretizer(Step):
    def __init__(self, num_atts, bin_size, encode='ordinal', strategy='kmeans'):
        """
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param bin_size: list of integer, each integer represents the number of bins to categorize the corresponding numerical attribute.
        :param encode: same parameter with sklearn KBinsDiscretizer
        :param strategy: same parameter with sklearn KBinsDiscretizer
        """
        self.focus_atts = num_atts
        self.bin_size = bin_size
        self.encode = encode
        self.strategy = strategy

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        for idx, ai in enumerate(self.focus_atts):
            fitted_step[ai] = KBinsDiscretizer(n_bins=self.bin_size[idx], encode=self.encode, strategy=self.strategy).fit(np.array(df[ai]).reshape(-1, 1))
        self.fitted_step = fitted_step
        return self

    def apply(self, df):
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "SK_Discretizer"

    def abbr_name(self):
        return "DS"

    def step_name(self):
        return "Categorizer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class SK_Binarizer(Step):
    def __init__(self, num_atts, bin_thresholds, copy=True):
        """
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param bin_thresholds: list of float, each float represents the value to binarize the corresponding numerical attributes.
                               Values below or equal to this threshold are replaced by 0, above it by 1.
        :param copy: same parameter with sklearn Binarizer
        """
        self.focus_atts = num_atts
        self.bin_thresholds = bin_thresholds
        self.copy = copy

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        for idx, ai in enumerate(self.focus_atts):
            fitted_step[ai] = Binarizer(threshold=self.bin_thresholds[idx], copy=self.copy).fit(np.array(df[ai]).reshape(-1, 1))
        self.fitted_step = fitted_step
        return self

    def apply(self, df):
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "SK_Binarizer"

    def abbr_name(self):
        return "BI"

    def step_name(self):
        return "Categorizer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':

    data = pd.read_csv("../../data/german.csv")
    # cur_o = SK_Discretizer(["month", "age"], [2, 3])
    cur_o = SK_Binarizer(["month", "age"], [24, 30])

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_"+cur_o.name()+".csv", index=False)
