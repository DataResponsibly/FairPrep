"""
    Classes to encode the string values for categorical attributes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pipeline.step import Step

class SK_OrdinalEncoder(Step):
    def __init__(self, cate_atts, sort_label, sort_positive_value):
        """
        :param cate_atts: list of str, each str represents the name of a categorical attribute in above data.
        :param sort_label: str, name of the target varible to determine the order of ordinal encodings.
        :param sort_positive_value: str, value of the target varible to determine the order of ordinal encodings.

        """

        self.focus_atts = cate_atts
        self.sort_label = sort_label
        self.sort_positive_value = sort_positive_value

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        for idx, ai in enumerate(self.focus_atts):
            value_counts = {}
            for vi in df[ai].unique():
                value_counts[vi] = df[(df[ai] == vi) & (df[self.sort_label] == self.sort_positive_value)].shape[0]
            value_orders = sorted(value_counts.keys(), key=lambda x: value_counts[x])
            fitted_step[ai] = OrdinalEncoder(categories=[value_orders]).fit(np.array(df[ai]).reshape(-1, 1))
        self.fitted_step = fitted_step
        return self

    def apply(self, df):
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "SK_OrdinalEncoder"

    def abbr_name(self):
        return "OE"

    def step_name(self):
        return "Encoder"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return True

    def fit_only_on_train(self):
        return True


class OneHotEncoder(Step):
    def __init__(self, cate_atts):
        """
        :param cate_atts: list of str, each str represents the name of a categorical attribute in above data.
        """
        self.focus_atts = cate_atts

    def fit(self, df):
        pass

    def apply(self, df):
        after_df = pd.get_dummies(df, columns=self.focus_atts, prefix_sep='=')
        return after_df

    def name(self):
        return "OneHotEncoder"

    def abbr_name(self):
        return "HE"

    def step_name(self):
        return "Encoder"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return True

    def fit_only_on_train(self):
        return False

# That can only apply in SensitiveAttributeEncoder
class MappingEncoder(Step):
    def __init__(self, focus_atts, mapping_dict):
        """ Encode sensitive attribute and target feature through mapping string to numberical values according to the input dictionary.
        :param focus_atts: list of str, each str represents the name of an attribute.
        :param mapping_dict: dict, key is the value (str) of the attribute in focus_atts, value is int that encode the value. E.g. {"female": 0, "male": 1} for the values of attribute gender.
        """

        self.focus_atts = focus_atts
        self.mapping_dict = mapping_dict

    def fit(self, df):
        pass


    def apply(self, df):
        after_df = df.copy()
        for si in self.focus_atts:
            after_df[si] = after_df[si].apply(lambda x: self.mapping_dict[x])
        return after_df

    def name(self):
        return "MappingEncoder"

    def abbr_name(self):
        return "ME"

    def step_name(self):
        return "SpecialEncoder"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return True

    def fit_only_on_train(self):
        return False


if __name__ == '__main__':
    data = pd.read_csv("../../data/german_AIF.csv")

    cur_o = SK_OrdinalEncoder(["sex"], "credit", "good")
    # cur_o = OneHotEncoder(["sex", "credit", "age"])

    # cur_o = MappingEncoder(["sex", "credit"], {"female": 0, "male": 1, "good": 1, "bad": 0})

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_AIF_"+cur_o.name()+".csv", index=False)
