"""
    Class to run the pipeline

"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from pipeline.preprocess.splitters import *
from pipeline.preprocess.samplers import *
from pipeline.preprocess.imputers import *
from pipeline.preprocess.scalers import *
from pipeline.preprocess.categorizers import *
from pipeline.preprocess.encoders import *
from pipeline.preprocess.fair_preprocessors import *
from pipeline.model.classifiers import *
from pipeline.model.fair_classifiers import *
from pipeline.postprocess.fair_postprocessors import *

# for integrity check of user inputs
SUPPORT_STEPS = {"Splitter": "Splitter", "Sampler": "Sampler", "Imputer": "Imputer", "Scaler": "Scaler",
                 "Categorizer": "Categorizer", "Encoder": "Encoder", "SensitiveEncoder": "SensitiveAttEncoder",
                 "FairPreprocessor": "AIF_", "model": "SK_OPT_", "FairPostprocessor": "AIF_Postprocessing"}

ALL_STEPS = ["RandomSplitter", "BalanceTargetSplitter",
             "RandomSampler", "BalancePopulationSampler",
             "DropNAImputer", "ModeImputer", "DatawigImputer",
             "SK_StandardScaler", "SK_MinMaxScaler",
             "SK_Discretizer", "SK_Binarizer",
             "SK_OrdinalEncoder", "SK_OneHotEncoder",
             "CustomCateAttsEncoder",
             "AIF_Reweighing", "AIF_DIRemover",
             "SK_LogisticRegression", "SK_DecisionTree", "OPT_LogisticRegression", "OPT_DecisionTree", "AIF_AdversarialDebiasing",
             "AIF_EqOddsPostprocessing", "AIF_CalibratedEqOddsPostprocessing"]
PRINT_SPLIT = "\n===================================================\n"

def init_input_steps(step_tuple, input_df):
    step_str = step_tuple[0] + '(input_df, '
    for pi in step_tuple[1:]:
        if isinstance(pi, str):
            step_str += "'"+pi + "', "
        else:
            step_str += str(pi) + ", "
    end_idx = step_str.rfind(", ")
    step_str = step_str[0:end_idx] + step_str[end_idx:].replace(", ", ")")
    print(step_str)
    return eval(step_str)


class FairnessLabelPipeline():
    def __init__(self, data_file_name, target_col, target_positive_values, sensitive_atts=[], protected_values={}, sep_flag=None, na_mark=None):
        """

        :param data_file_name: str, file name that stores the data.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_values: list of str, each str is the value of the target variable in above data that represents the positive outcome.
        :param sensitive_atts: list, stores the user specified sensitive attributes in above data. Optional.
        :param protected_values: dict, stores the user-specified protected values for above sensitive attributes. Optional.
                                Key is the name in sensitive_atts, value is a list of str, representing the values of the attribute.
                                 Order should mapping to the order in the sensitive_atts.
        """
        # print(os.path.realpath(data_file_name))
        if not os.path.exists(os.path.realpath(data_file_name)):
            print("The data you specified doesn't exist!")
            raise ValueError
        if ".csv" not in data_file_name:
            print("The data you specified is not valid! Only support .csv file.")
            raise ValueError
        data_name = data_file_name.replace(".csv", "")
        self.data_name = data_name[data_name.rfind("/")+1:]
        if sep_flag:
            df = pd.read_csv(data_file_name, sep=sep_flag)
        else: # default ',' separated data
            df = pd.read_csv(data_file_name)
        if not df.shape[0]:
            print("Uploaded data is empty!")
            raise ValueError

        # integrity check for target_col
        if target_col is None or target_positive_values is None:
            print("Need to specify target_col and target_positive_value!")
            raise ValueError

        if target_col not in df.columns:
            print("Need to specify a valid target attribute to be predicted!")
            raise ValueError
        target_values = df[target_col].unique()
        if len(target_values) != 2:
            print("Only support binary target feature now!")
            raise ValueError
        if len(set(target_positive_values).intersection(target_values)) == 0:
            print("Need to specify a valid target positive value!")
            raise ValueError
        self.target_col = target_col
        self.target_positive_values = target_positive_values

        # integrity check for sensitive_atts and protected_values
        input_indicator = sum([len(x)== 0 for x in [sensitive_atts, protected_values]])
        if input_indicator == 0: # both are specified
            if len(sensitive_atts) != len(protected_values):
                print("Different size of input sensitive attributes and protected values!")
                raise ValueError
            if sum([len(set(protected_values[x]).difference(df[x].unique())) > 0 for x in protected_values]) > 0:
                print("Some specified protected values do not appear in the column specified in sensitive_atts!")
                raise ValueError
        elif input_indicator == 1: # one of parameter is empty
            print("Need to specify both sensitive_atts and protected_values!")
            raise ValueError
        else: # both are empty
            # TODO: add auto-generation for the below two variables: sensitive_atts and protected_values
            # for adult only
            sensitive_atts = ["sex", "race"]
            protected_values = {"sex": ["Female"], "race": ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]}

        self.sensitive_atts = sensitive_atts
        # self.protected_values = protected_values
        self.zero_mapping = {target_col: [x for x in target_values if x not in target_positive_values]}
        self.zero_mapping.update(protected_values)
        if na_mark:
            self.na_mark = na_mark
        else:
            self.na_mark = None

        # refer numerical and categorical attributes first
        # DataFrame.describe() usually returns 8 rows.
        if df.describe().shape[0] == 8:
            num_atts = set(df.describe().columns)
        # DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
        else:
            num_atts = set()

        cate_atts = set(df.columns).difference(num_atts)
        if self.target_col in cate_atts:
            cate_atts.remove(self.target_col)
        if self.target_col in num_atts:
            num_atts.remove(self.target_col)
        for si in sensitive_atts:
            cate_atts.remove(si)
        self.num_atts = list(num_atts)
        self.cate_atts = list(cate_atts)
        self.df = df

        # record the sensitive attributes and target variable value mapping
        sensi_target_value_mapping = {}
        for atti in [self.target_col] + self.sensitive_atts:
            atti_values = list(df[atti].unique())
            pro_values = self.zero_mapping[atti]
            if len(pro_values) > 1:
                pro_value_str = pro_values[0] + "&more"
            else:
                pro_value_str = pro_values[0]
            other_values = list(set(atti_values).difference(pro_values))
            if len(other_values) > 1:
                other_value_str = other_values[0] + "&more"
            else:
                other_value_str = other_values[0]
            sensi_target_value_mapping[atti] = {0: pro_value_str, 1: other_value_str}
        # print(sensi_target_value_mapping)
        self.sensi_target_value_mapping = sensi_target_value_mapping
        self.pipeline_id = self.data_name[:2]

    def init_necessary_steps(self, step_flag, apply_df, input_weights=[]):
        # initialize the necessary steps
        if step_flag == "Imputer":
            return DropNAImputer(apply_df, na_mark=self.na_mark)
        # elif step_flag == "Scaler":
        #     return SK_StandardScaler(apply_df, list(self.num_atts))
        elif step_flag == "Encoder":
            return SK_OneHotEncoder(apply_df, list(self.cate_atts))
        elif step_flag == "SensitiveEncoder":
            return CustomCateAttsEncoder(apply_df, self.sensitive_atts+[self.target_col], self.zero_mapping)
        else:
            return OPT_LogisticRegression(apply_df, self.target_col, instance_weights=input_weights)

    def print_necessary_steps(self):
        # for printout and efficiency
        return {"Imputer": ("DropNAImputer", "?"),
                # "Scaler": ("SK_StandardScaler", list(self.num_atts)),
                "Encoder": ("SK_OneHotEncoder", list(self.cate_atts)),
                "SensitiveEncoder": ("CustomCateAttsEncoder", self.sensitive_atts+[self.target_col], self.zero_mapping),
                "model": ("OPT_LogisticRegression", self.target_col)}


    def run_pipeline(self, steps, return_test=True, output_interdata=False):
        """

        :param df: pandas dataframe, the data on which steps are applied.
        :param steps: list of classes that represent the steps user want to perform on the above data.
                      Supported steps are listed in STEPS.md.
        :return:  two pandas dataframes: before_data and after_data. after_data is the data after applied the input steps.

        """

        if not steps:
            print("Require list of steps as input!")
            raise ValueError

        if len(steps) < len(SUPPORT_STEPS):
            print("Missing some input steps! Required steps are listed in the order below.\n"+" ".join(SUPPORT_STEPS.keys()))
            raise ValueError
        if sum([len(set(x[0]).intersection(list(SUPPORT_STEPS.values())[idx]))==0 for idx, x in enumerate(steps) if x is not None]) > 0:
            print("Some input steps are not supported!")
            raise ValueError
        if sum([x[0] not in ALL_STEPS for x in steps if x is not None]) > 0:
            print("Some input steps don't include enough parameters!")
            raise ValueError
        if sum([len(x) <=1 for x in steps if x is not None]) > 0:
            print("Some input steps don't include enough parameters!")
            raise ValueError

        if steps[-1] is not None and len(steps[0][1])==2: # run fair-postprocessor, then require validation set
            print("FairPostprocessor requires a validation set! Specify through split_ratio in Splitter!")
            raise ValueError

        self.pipeline_id = "_".join([self.pipeline_id]+[str(x[0]) for x in steps if x is not None])

        support_step_names = list(SUPPORT_STEPS.keys())

        # split the data into separated datasets for train, [validation], and test
        if steps[0] is None: # default splitter
            if steps[-1] is not None: # train, validation and test data
                cur_splitter = BalanceTargetSplitter(self.df, [0.5, 0.3, 0.2], self.target_col)
            else: # train and test data
                cur_splitter = BalanceTargetSplitter(self.df, [0.7, 0.3], self.target_col)
        else:
            cur_splitter = init_input_steps(steps[0], self.df)
        after_data = cur_splitter.apply(self.df)
        after_data = list(after_data)
        if output_interdata:
            self.save_inter_data(after_data, cur_splitter.get_name())
        print("Done "+support_step_names[0]+PRINT_SPLIT)
        # record the before data to output
        before_data = [x for x in after_data]


        # run sampler on train
        if steps[1] is not None:
            for idx_df, cur_df in enumerate(after_data):
                cur_sampler = init_input_steps(steps[1], cur_df)
                after_data[idx_df] = cur_sampler.apply(cur_df)
            if output_interdata:
                self.save_inter_data(after_data, cur_sampler.get_name(), steps[:1])
            print("Done "+support_step_names[1]+PRINT_SPLIT)


        # run the preprocess steps: "Imputer", "Scaler", "Categorizer" that fit on train and apply on others
        for idx, step_i in enumerate(steps[2:5]):
            idx = idx + 2
            step_i_key = support_step_names[idx]
            # fit on train data
            if step_i is None:
                if step_i_key in list(self.print_necessary_steps().keys()):  # add default operation for necessary steps
                    step_i = self.init_necessary_steps(step_i_key, after_data[0])
                else:  # skip the step
                    continue
            else:  # user-specified step
                step_i = init_input_steps(steps[idx], after_data[0])
            # apply on train, validation and test data
            for idx_df, cur_df in enumerate(after_data):
                after_data[idx_df] = step_i.apply(cur_df)

            if output_interdata:
                self.save_inter_data(after_data, step_i.get_name(), steps[:idx])
            print("Done " + support_step_names[idx] + PRINT_SPLIT)

        # run the preprocess steps: "Encoder"
        # fit and apply on the same data
        after_data, encoder_name = self.run_encoder(steps[5], after_data)
        if output_interdata:
            self.save_inter_data(after_data, encoder_name, steps[:5])
        print("Done " + support_step_names[5] + PRINT_SPLIT)

        # run the preprocess steps: "SensitiveAttEncoder"
        # fit and apply on the same data
        if steps[6] is None:
            for idx_df, cur_df in enumerate(after_data):
                cur_sensi_encoder = self.init_necessary_steps("SensitiveEncoder", cur_df)
                after_data[idx_df] = cur_sensi_encoder.apply(cur_df)
        else: # user-specified sensitive encoder
            for idx_df, cur_df in enumerate(after_data):
                cur_sensi_encoder = init_input_steps(steps[6], cur_df)
                after_data[idx_df] = cur_sensi_encoder.apply(cur_df)
        if output_interdata:
            self.save_inter_data(after_data, cur_sensi_encoder.get_name(), steps[:6])
        print("Done " + support_step_names[6] + PRINT_SPLIT)

        # run the preprocess steps: "FairPreprocessor"
        # fit and apply on the same data
        if steps[7] is not None:
            weights = [[0 for _ in range(x.shape[0])] for x in after_data]
            for idx_df, cur_df in enumerate(after_data):
                cur_fair_preprossor = init_input_steps(steps[7], cur_df)
                if "AIF_Reweighing" in cur_fair_preprossor.get_name():  # special heck for methods updating sample weight
                    after_data[idx_df], weights[idx_df] = cur_fair_preprossor.apply(cur_df)
                else:
                    after_data[idx_df] = cur_fair_preprossor.apply(cur_df)
            if output_interdata:
                self.save_inter_data(after_data, cur_fair_preprossor.get_name(), steps[:7])
            print("Done " + support_step_names[7] + PRINT_SPLIT)

            # after fair-preprocess, rerun encoder
            after_data, encoder_name = self.run_encoder(steps[5], after_data)
            if output_interdata:
                self.save_inter_data(after_data, encoder_name+"_prep", steps[:5])
            print("Done " + support_step_names[5] + " for fair preprocessor "+PRINT_SPLIT)

        # run model step
        # fit on train data
        if steps[8] is None:
            if weights:
                cur_model = self.init_necessary_steps("model", after_data[0], input_weights=weights[0])
            else:
                cur_model = self.init_necessary_steps("model", after_data[0])
        else: # TODO: add the support for weight in user-specified models
            cur_model = init_input_steps(steps[8], after_data[0])

        # predict on train, validation and test data
        for idx_df, cur_df in enumerate(after_data):
            after_data[idx_df] = cur_model.apply(cur_df)
        if output_interdata:
            self.save_inter_data(after_data, cur_model.get_name(), steps[:8])
        print("Done " + support_step_names[8] + PRINT_SPLIT)

        # run fair postprocess step
        if steps[9] is not None:
            # encode first
            after_data, encoder_name = self.run_encoder(steps[5], after_data)
            if output_interdata:
                self.save_inter_data(after_data, encoder_name+"_post", steps[:5])
            print("Done " + support_step_names[5] + " for fair post processor " + PRINT_SPLIT)
            # fit on validation data
            cur_postprocessor = init_input_steps(steps[9], after_data[1])
            # predict on validation and test data
            for idx_df, cur_df in enumerate(after_data[1:]):
                after_data[idx_df+1] = cur_postprocessor.apply(cur_df)
            if output_interdata:
                self.save_inter_data(after_data, cur_postprocessor.get_name(), steps[:9])
            print("Done " + support_step_names[9] + PRINT_SPLIT)

        # transfer back to original values for encoded sensitive and target columns
        for idx, df_i in enumerate(after_data):
            for atti in [self.target_col]+ self.sensitive_atts:
                df_i[atti] = df_i[atti].apply(lambda x: self.sensi_target_value_mapping[atti][x])
            if "pred_" +self.target_col in df_i.columns:
                df_i["pred_" + self.target_col] = df_i["pred_" + self.target_col].apply(lambda x: int(x>=0.5))
                df_i["pred_" +self.target_col] = df_i["pred_" +self.target_col].apply(lambda x: self.sensi_target_value_mapping[self.target_col][x])

        if return_test: # only return the before and after of test data
            return before_data[-1], after_data[-1]
        else: # return all before and after data
            return before_data, after_data

    def run_encoder(self, encode_step_tuple, data_list):
        # run the preprocess steps: "Encoder"
        # fit and apply on the same data
        if len(self.cate_atts) > 0:
            for idx_df, cur_df in enumerate(data_list):
                if encode_step_tuple is None:  # default encoder
                    cur_encoder = self.init_necessary_steps(list(SUPPORT_STEPS.keys())[5], cur_df)
                else:

                    # check for user specified encoder that cover partial categorical atts
                    non_encoded_cate = set(self.cate_atts).difference(encode_step_tuple[1])
                    if non_encoded_cate:
                        cur_encoder = init_input_steps((encode_step_tuple[0], self.cate_atts), cur_df)
                    else:
                        cur_encoder = init_input_steps(encode_step_tuple, cur_df)
                data_list[idx_df] = cur_encoder.apply(cur_df)

            # check for different dimensions after encoding for validation and test set
            if len(data_list) > 2:
                for idx_df, cur_df in enumerate(data_list[1:]):
                    if cur_df.shape[1] != data_list[0].shape[1]:
                        for feature_i in set(data_list[0].columns).difference(cur_df.columns):
                            cur_df[feature_i] = 0.0
                        data_list[idx_df] = cur_df.copy()
            else: # check the dimensions for train and test set
                if data_list[0].shape[1] != data_list[1].shape[1]:
                    diff_features_1 = set(data_list[0].columns).difference(data_list[1].columns)
                    diff_features_2 = set(data_list[1].columns).difference(data_list[0].columns)
                    add_df = data_list[1].copy()
                    for feature_i in diff_features_1.union(diff_features_2):
                        if feature_i not in add_df.columns:
                            add_df[feature_i] = 0.0
                    data_list[1] = add_df.copy()

                    add_df = data_list[0].copy()
                    for feature_i in diff_features_1.union(diff_features_2):
                        if feature_i not in add_df.columns:
                            add_df[feature_i] = 0.0
                    data_list[0] = add_df.copy()

            return data_list, cur_encoder.get_name()
        else:
            return data_list, "None"

    def save_inter_data(self, input_dfs, step_name, pre_steps=[], path="data/inter_data/"):
        if len(input_dfs) == 2:
            suffix_names = ["train", "test"]
        else:
            suffix_names = ["train", "validation", "test"]

        for idx, df_i in enumerate(input_dfs):
            output_df_i = df_i.copy()
            df_path = os.path.realpath(path) + "/" + self.pipeline_id + "/" + suffix_names[idx] + "/"
            if not os.path.exists(df_path):
                os.makedirs(df_path)
            if pre_steps:
                pre_step_names = [x[0] for x in pre_steps if x is not None]+[step_name]
            else:
                pre_step_names = [step_name]

            for atti in [self.target_col]+ self.sensitive_atts:
                if not isinstance(output_df_i[atti].values[0], str):
                    output_df_i[atti] = output_df_i[atti].apply(lambda x: self.sensi_target_value_mapping[atti][x])
            if "pred_" +self.target_col in output_df_i.columns:
                output_df_i["pred_" + self.target_col] = output_df_i["pred_" + self.target_col].apply(lambda x: int(x >= 0.5))
                output_df_i["pred_" +self.target_col] = output_df_i["pred_" +self.target_col].apply(lambda x: self.sensi_target_value_mapping[self.target_col][x])

            output_name = df_path + "__".join([self.data_name, "after"]+[x[:x.find("@")] for x in pre_step_names]) + ".csv"
            print("!!!!!!!", suffix_names[idx], output_df_i.shape, "!!!!!!!")
            output_df_i.to_csv(output_name, index=False)
            print("Current "+suffix_names[idx]+" data after "+" ".join([x[:x.find("@")] for x in pre_step_names])+" \n Stored in ", output_name)
            print()


    

if __name__ == '__main__':
    # input_steps = [("BalanceTargetSplitter", [0.5, 0.3, 0.2], "income-per-year"),
    #                ("RandomSampler", 10000),  # sampler
    #                ("DropNAImputer", "?"),
    #                ("SK_StandardScaler", ["fnlwgt", "age"]),
    #                ("SK_Discretizer", ["fnlwgt", "age"], [2, 3]),
    #                ("SK_OneHotEncoder", ["workclass"]),  # encoder
    #                ("CustomCateAttsEncoder", ["sex", "race", "income-per-year"], {"sex": ["Female"], "race": ["Black"], "income-per-year": ["<=50K"]}),
    #                ("AIF_DIRemover", "income-per-year", "sex", 0.8),  # fair-preprocessor
    #                ("AIF_AdversarialDebiasing", "income-per-year", "sex"), # test Adversial learning
    #                ("AIF_CalibratedEqOddsPostprocessing", "income-per-year", "sex")  # fair-post-postprocessor
    #                ]
    # cur_pip = FairnessLabelPipeline(data_file, y_col, y_posi, sensitive_atts=sensi_atts, protected_values=sensi_pro_valus, na_mark="?")
    # before_test, after_test = cur_pip.run_pipeline(input_steps, return_test=True, output_interdata=True)

    # data_file = "../data/adult.csv"
    # y_col = "income-per-year"
    # y_posi = [">50K"]
    # na_symbol = "?"
    # sensi_atts = ["sex", "race"]
    # sensi_pro_valus = {"sex": ["Female"], "race": ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]}

    # data_file = "../data/mylsn_cleaned_2.csv"
    # y_col = "status"
    # y_posi = ["Ac"]
    # na_symbol = "N/A"
    # sensi_atts = ["sex", "race"]
    # sensi_pro_valus = {"sex": ["female"], "race": ["black", "hispanic", "native-american", "asian"]}

    #
    #  input_steps = [("BalanceTargetSplitter", [0.7, 0.3], y_col),  # splitter
    #               None,  # sampler
    #               ("DropNAImputer", "?"),  # imputer
    #               None,  # scaler
    #               ("SK_Discretizer", ["age"], [3]),  # categorizer
    #               None,  # encoder
    #               None,  # sensitive att and target encoder
    #               None,  # fair-preprocessor
    #               ("OPT_LogisticRegression", y_col),  # model
    #               None  # fair-post-postprocessor
    #               ]

    # debias_focus_att = "race"
    # input_steps = [("BalanceTargetSplitter", [0.7, 0.3], y_col),
    #               ("RandomSampler", 5000),  # sampler
    #               None,  # imputer
    #               None,  # scaler
    #               None,
    #               None,  # encoder
    #               None,
    #               None, #("AIF_DIRemover", y_col, debias_focus_att, 1.0),  # fair-preprocessor
    #               ("SK_LogisticRegression", y_col),  # model
    #               None  # fair-post-postprocessor
    #               ]

    data_file = "../data/german_AIF.csv"
    y_col = "credit"
    y_posi = ["good"]
    sensi_atts = ["age", "sex"]
    sensi_pro_valus = {"age": ["young"], "sex": ["female"]}
    debias_focus_att = "age"
    fair_steps = [("BalanceTargetSplitter", [0.7, 0.3], y_col),
                  None,  # sampler
                  None,  # ("ModeImputer", [], ["workclass"], "?"), # imputer
                  None,  # scaler
                  None,  # categorizer
                  None,  # encoder
                  None,
                  ("AIF_Reweighing", y_col, debias_focus_att),  # fair-preprocessor
                  None,  # ("OPT_LogisticRegression", y_col), # model
                  None  # fair-post-postprocessor
                  ]

    cur_pip = FairnessLabelPipeline(data_file, y_col, y_posi, sensitive_atts=sensi_atts,
                                    protected_values=sensi_pro_valus)
    before_test, after_test = cur_pip.run_pipeline(fair_steps, return_test=True, output_interdata=True)

