"""
    Class of fairness preprocessing interventions
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing as Reweighing
from aif360.algorithms.preprocessing import LFR as LFR
from aif360.algorithms.preprocessing import DisparateImpactRemover
from FairPrep.step import Step

class NoFairPreprocessor(Step):
    def __init__(self):
        self.fitted_step = None

    def fit(self, df):
        pass

    def apply(self, df):
        return df

    def name(self):
        return "NoPreprocessor"

    def abbr_name(self):
        return "NP"

    def step_name(self):
        return "FairPreprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return True

    def fit_only_on_train(self):
        return False

class AIF_Reweighing(Step):

    def __init__(self, target_col, sensitive_att):
        """
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data.

        """
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name

        self.fitted_step = None

    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        self.fitted_step = Reweighing([{self.sensitive_att: 0}], [{self.sensitive_att: 1}]).fit(aif_df)
        return self

    def apply(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        after_aif_df = self.fitted_step.transform(aif_df)
        # TODO: double check whether to return weights
        preprocessed_weights = after_aif_df.instance_weights

        after_df, _ = after_aif_df.convert_to_dataframe()
        
        return after_df

    def name(self):
        return "Reweighing"

    def abbr_name(self):
        return "RW"

    def step_name(self):
        return "FairPreprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_DIRemover(Step):
    def __init__(self, target_col, sensitive_att, repair_level):
        """
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data.

        """
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name

        self.fitted_step = None

        if repair_level is None or not isinstance(repair_level, float):
            print("Input repair_level is not valid! Should be float within [0,1]!")
            raise ValueError
        else:
            if repair_level < 0 or repair_level > 1:
                print("Input repair_level is not valid! Should be float within [0,1]!")
                raise ValueError

        self.repair_level = repair_level


    def fit(self, df):
        # TODO: Rewrite the AIF DIRemover to afford the separation of train, validation, and test.
        # aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        # self.fitted_step = DisparateImpactRemover(repair_level=self.repair_level, sensitive_attribute=self.sensitive_att).fit(aif_df)
        #
        pass

    def apply(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        after_aif_df = DisparateImpactRemover(repair_level=self.repair_level, sensitive_attribute=self.sensitive_att).fit_transform(aif_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        return after_df

    def name(self):
        return "DIRemover"

    def abbr_name(self):
        return "RI"

    def step_name(self):
        return "FairPreprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class AIF_LFR(Step):

    def __init__(self, target_col, sensitive_att):
        """ NOTE: very sensitive to input data, refer the example in AIF 360 for this preprocessor
        :param target_col: str, the name of the target variable in above data.
        :param sensitive_att: str, the name of a sensitive attribute in above data.

        """
        # TODO: fix the bug of LFR for not returning categorical atts
        # TODO: experiment with the same data used by AIF360 tutorial to compare whether the categorical atts are returned
        self.sensitive_att = sensitive_att
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name

        self.fitted_step = None

    def fit(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        self.fitted_step = LFR([{self.sensitive_att: 0}], [{self.sensitive_att: 1}]).fit(aif_df)
        return self

    def apply(self, df):
        # wrap the input dataframe with AIF 360 object "BinaryLabelDataset"
        aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
        after_aif_df = self.fitted_step.transform(aif_df)

        after_df, _ = after_aif_df.convert_to_dataframe()
        return after_df

    def name(self):
        return "LFR"

    def abbr_name(self):
        return "LR"

    def step_name(self):
        return "FairPreprocessor"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_pre_encoded.csv")
    # name_mapping = {"female": 0, "male": 1, "young": 0, "old": 1, "bad": 0, "good": 1}
    # for atti in ["credit", "sex", "age"]:
    #     data[atti] = data[atti].apply(lambda x: name_mapping[x])
    #
    # data.to_csv("../../data/german_pre_encoded.csv")

    cur_o = AIF_Reweighing("credit", "sex")
    # cur_o = AIF_LFR("credit", "sex") # TODO: bug not working for this dataset, test another dataset
    # cur_o = AIF_DIRemover("credit", "sex", 0.8)
    cur_o.fit(data)
    after_data = cur_o.apply(data)
    # # for Reweighing
    # after_data, new_weights = cur_o.apply(data)
    after_data.to_csv("../../data/german_after_"+cur_o.name()+"_AAA.csv", index=False)

    # print(cur_o.get_name())
    # for Reweighing
    # print(len(new_weights))
    # print(new_weights)