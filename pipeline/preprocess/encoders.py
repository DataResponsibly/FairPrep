"""
    Classes to encode the string values for categorical attributes.
"""
"""
    Classes to discretize numerical attributes into categorical attributes.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from pipeline.preprocess.preprocessor import Preprocessor

class SK_OrdinalEncoder(Preprocessor):
    def __init__(self, df, cate_atts, sort_label, sort_positive_value):
        """
        :param df: pandas dataframe, stores the data to fit the encoder.
        :param cate_atts: list of str, each str represents the name of a categorical attribute in above data.
        :param encode_order_dict: dict, key (str) represents the name of categorical attribute, value is a list of str, representing the ordered categories of each each categorical attribute.
        :param sort_label: str, name of the target varible to determine the order of ordinal encodings.
        """
        cur_step = {}
        for ci in cate_atts:
            value_counts = {}
            for vi in df[ci].unique():
                value_counts[vi] = df[(df[ci] == vi) & (df[sort_label] == sort_positive_value)].shape[0]
            value_orders = sorted(value_counts.keys(), key=lambda x: value_counts[x])
            cur_step[ci] = OrdinalEncoder(categories=[value_orders])

        self.sort_label = sort_label
        self.sort_positive_value = sort_positive_value
        super().__init__("@".join(["OrdinalEncoder"]+cate_atts), df, step=cur_step, focus_atts=cate_atts, fit_flag=True)


class SK_OneHotEncoder(Preprocessor):
    def __init__(self, df, cate_atts):
        """
        :param df: pandas dataframe, stores the data to fit the encoder.
        :param cate_atts: list of str, each str represents the name of a categorical attribute in above data.
        :param encode_order_dict: dict, key (str) represents the name of categorical attribute, value is a list of str, representing the ordered categories of each each categorical attribute.
        :param sort_label: str, name of the target variable to determine the order of ordinal encodings.
        """
        # TODO: fix the bug that sklearn one-hot encoder change the dimension
        # cur_step = {}
        # for ci in cate_atts:
        #     cur_step[ci] = OneHotEncoder()
        super().__init__("@".join(["OneHotEncoder"]+cate_atts), df, step=None, focus_atts=cate_atts, fit_flag=False)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned encoder.
        :return: pandas dataframe, stores the data after encode.
        """
        after_df = pd.get_dummies(df, columns=self.focus_atts, prefix_sep='=')
        # after_df = df[list(set(df.columns).difference(self.focus_atts))]
        # for ci in self.focus_atts:
        #     ci_encode_array = self.step[ci].transform(np.array(df[ci]).reshape(-1, 1)).toarray()
        #     ci_encode_df = pd.DataFrame(ci_encode_array, columns=[ci+"="+x for x in self.step[ci].categories_[0]])
        #     after_df = pd.concat([after_df, ci_encode_df], axis=1)
        return after_df

class CustomCateAttsEncoder(Preprocessor):
    def __init__(self, df, sensitive_atts, protected_values):
        """ To encode sensitive attribute and target feature.
        :param df: pandas dataframe, stores the data to fit the encoder.
        :param sensitive_atts: list of str, each str represents the name of a sensitive attribute.
        :param protected_values: dict, key is the str in sensitive_atts, value is a list of str, each str represent the protected values for the key sensitive attribute.
        """
        super().__init__("@".join(["SensitiveAttEncoder"]+sensitive_atts), df, step=None, focus_atts=sensitive_atts, fit_flag=False)
        for x in sensitive_atts:
            if sum([vi not in df[x].unique() for vi in protected_values[x]]) > 0:
                print("Some input values of sensitive attribute ", x, " are not valid!")
                raise ValueError
        self.protected_values = protected_values

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the encoder.
        :return: pandas dataframe, stores the data after encode.
        """
        after_df = df.copy()
        for si in self.focus_atts:
            after_df[si] = after_df[si].apply(lambda x: int(x not in self.protected_values[si]))
        return after_df



if __name__ == '__main__':
    data = pd.read_csv("../data/train/adult__Imputer.csv")
    # data = pd.read_csv("../../data/adult_pre_RandomSampler_1000.csv")
    # data = pd.read_csv("../../data/adult_pre_SensitiveAttsEncoder_sex_race_income-per-year.csv")
    # cur_o = SK_OrdinalEncoder(data, ["sex", "race"], "income-per-year", ">50K")
    cur_o = SK_OneHotEncoder(data, ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"])
    # cur_o = CustomCateAttsEncoder(data, ["sex", "race", "income-per-year"], {"sex": ["Female"], "race": ["Black"], "income-per-year": ["<=50K"]})

    after_data = cur_o.apply(data)
    after_data.to_csv("../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())