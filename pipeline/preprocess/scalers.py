"""
    Classes to scale data.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pipeline.preprocess.preprocessor import Preprocessor

class SK_StandardScaler(Preprocessor):

    def __init__(self, df, num_atts, copy=True, with_mean=True, with_std=True):
        """
        :param df: pandas dataframe, stores the data to fit the scaler.
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param copy: same parameter with sklearn StandardScaler
        :param with_mean: same parameter with sklearn StandardScaler
        :param with_std: same parameter with sklearn StandardScaler
        """
        cur_step = {}
        for ai in num_atts:
            cur_step[ai] = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

        super().__init__("@".join(["StandardScaler"]+num_atts), df, step=cur_step, focus_atts=num_atts, fit_flag=True)


class SK_MinMaxScaler(Preprocessor):
    def __init__(self, df, num_atts, feature_range=(0, 1), copy=True):
        """
        :param df: pandas dataframe, stores the data to fit the scaler.
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param feature_range: same parameter with sklearn MinMaxScaler
        :param copy: same parameter with sklearn MinMaxScaler
        """
        cur_step = {}
        for ai in num_atts:
            cur_step[ai] = MinMaxScaler(feature_range=feature_range, copy=copy)
        super().__init__("@".join(["MinMaxScaler"]+num_atts), df, step=cur_step, focus_atts=num_atts, fit_flag=True)


if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_RandomSampler_1000.csv")
    cur_o = SK_StandardScaler(data, ["fnlwgt", "age"])
    # cur_o = SK_MinMaxScaler(data, ["fnlwgt", "age"])

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())