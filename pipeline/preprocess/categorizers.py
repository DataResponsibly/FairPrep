"""
    Classes to discretize numerical attributes into categorical attributes.
"""
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from pipeline.preprocess.preprocessor import Preprocessor


class SK_Discretizer(Preprocessor):
    def __init__(self, df, num_atts, bin_size, encode='ordinal', strategy='kmeans'):
        """
        :param df: pandas dataframe, stores the data to fit the discretizer.
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param bin_size: list of integer, each integer represents the number of bins to categorize the corresponding numerical attribute.
        :param encode: same parameter with sklearn KBinsDiscretizer
        :param strategy: same parameter with sklearn KBinsDiscretizer
        """
        cur_step = {}
        for idx, ai in enumerate(num_atts):
            cur_step[ai] = KBinsDiscretizer(n_bins=bin_size[idx], encode=encode, strategy=strategy)
        self.bin_size = bin_size
        super().__init__("@".join([strategy+"Categorizer"]+num_atts), df, step=cur_step, focus_atts=num_atts, fit_flag=True)


class SK_Binarizer(Preprocessor):
    def __init__(self, df, num_atts, bin_thresholds, copy=True):
        """
        :param df: pandas dataframe, stores the data to fit the binarizer.
        :param num_atts: list of str, each str represents the name of a numerical attribute in above data.
        :param bin_thresholds: list of float, each float represents the value to binarize the corresponding numerical attributes.
                               Values below or equal to this threshold are replaced by 0, above it by 1.
        :param copy: same parameter with sklearn Binarizer
        """
        cur_step = {}
        for idx, ai in enumerate(num_atts):
            cur_step[ai] = Binarizer(threshold=bin_thresholds[idx], copy=copy)

        self.bin_thresholds = bin_thresholds
        super().__init__("@".join(["BinaryCategorizer"]+num_atts), df, step=cur_step, focus_atts=num_atts, fit_flag=True)


if __name__ == '__main__':
    data = pd.read_csv("../../data/adult.csv")
    cur_o = SK_Discretizer(data, ["fnlwgt", "age"], [2, 3])
    # cur_o = SK_Binarizer(data, ["fnlwgt", "age"], [100000, 30])

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())