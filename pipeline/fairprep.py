"""
    Class to run the pipeline

"""


import os
import numpy as np
import pandas as pd

from time import time
from pathlib import Path
from datetime import datetime
from itertools import islice


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

import json

PRINT_SPLIT = "\n===================================================\n"
# for integrity check of user inputs
PREREQUISITE_STEPS = {"Splitter": [], "Sampler": [], "Imputer": ["Splitter"], "Scaler": ["Imputer"],
                    "Categorizer": ["Imputer"], "Encoder": ["Imputer"], "SpecialEncoder": ["Imputer"],
                    "FairPreprocessor": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"],
                    "Model": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SensitiveAttEncoder"],
                    "FairPostprocessor": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"]}



class FairPipeline():
    def __init__(self, data_file_name, target_col, target_positive_values, sensitive_atts, value_mapping, sep_flag=None, na_mark=None, verbose=True):
        """

        :param data_file_name: str, file name that stores the data.
        :param *fixed_random_seed: int, the fixed random seed that will be used through the pipeline.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_values: list of str, each str is the value of the target variable in above data that represents the positive outcome.
        :param sensitive_atts: list, stores the user specified sensitive attributes in above data. Optional.
        :param protected_values: dict, stores the user-specified protected values for above sensitive attributes. Optional.
                                Key is the name in sensitive_atts, value is a list of str, representing the values of the attribute.
                                 Order should mapping to the order in the sensitive_atts.
        """

        # TODO: move integrity check to a separate function
        self.data_file_name = data_file_name
        self.target_col = target_col
        self.target_positive_values = target_positive_values
        self.sensitive_atts = sensitive_atts
        self.value_mapping = value_mapping
        self.sep_flag = sep_flag
        self.na_mark = na_mark
        self.verbose = verbose

        self.data_name = None
        self.raw_data = None

        self.attributes = None
        self.metadata_desp = None
        self.num_atts = None
        self.cate_atts = None
        self.pred_target = "pred_" + target_col

        self.log_path = 'logs/'
        self.exec_timestamp = self.generate_timestamp()
        self.log_dir_name = None

    def validate_input_parameters(self):
        if not os.path.exists(os.path.realpath(self.data_file_name)):
            print("The data you specified doesn't exist!")
            raise ValueError
        if ".csv" not in self.data_file_name:
            print("The data you specified is not valid! Only support .csv file.")
            raise ValueError
        data_name = self.data_file_name.replace(".csv", "")
        self.data_name = data_name[data_name.rfind("/")+1:]

        if self.sep_flag:
            df = pd.read_csv(self.data_file_name, sep=self.sep_flag)
        else: # default ',' separated data
            df = pd.read_csv(self.data_file_name)

        if not df.shape[0]:
            print("Uploaded data is empty!")
            raise ValueError

        # np.random.seed(fixed_random_seed)
        # self.fixed_random_seed = fixed_random_seed

        # integrity check for target_col
        if self.target_col is None or self.target_positive_values is None:
            print("Need to specify target_col and target_positive_value!")
            raise ValueError

        if self.target_col not in df.columns:
            print("Need to specify a valid target attribute to be predicted!")
            raise ValueError
        target_values = df[self.target_col].unique()
        if len(target_values) != 2:
            print("Only support binary target feature now!")
            raise ValueError
        if len(set(self.target_positive_values).intersection(target_values)) == 0:
            print("Need to specify a valid target positive value!")
            raise ValueError

        # TODO: update the below code
        # # integrity check for sensitive_atts and protected_values
        # input_indicator = sum([len(x)== 0 for x in [self.sensitive_atts, self.protected_values]])
        # if input_indicator == 0: # both are specified
        #     if len(self.sensitive_atts) != len(self.protected_values):
        #         print("Different size of input sensitive attributes and protected values!")
        #         raise ValueError
        #     if sum([len(set(self.protected_values[x]).difference(df[x].unique())) > 0 for x in self.protected_values]) > 0:
        #         print("Some specified protected values do not appear in the column specified in sensitive_atts!")
        #         raise ValueError
        # elif input_indicator == 1: # one of parameter is empty
        #     print("Need to specify both sensitive_atts and protected_values!")
        #     raise ValueError
        # else: # both are empty
        #     print("Need to specify both sensitive_atts and protected_values!")
        #     raise ValueError

        self.raw_data = df
        self.attributes = df.columns
        return self

    def get_meta_information(self):
        # infer the numerical and categorical attributes first
        if self.raw_data.describe().shape[0] == 8:# DataFrame.describe() usually returns 8 rows.
            num_atts = set(self.raw_data.describe().columns)
        else:# DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
            num_atts = set()

        cate_atts = set(self.attributes).difference(num_atts)


        self.metadata_desp = {"size": self.raw_data.shape[0], "features": self.raw_data.shape[1],
                              "categorical features": list(cate_atts), "numerical features": list(num_atts)}
        # record the domain of the attributes
        feature_domain = {}
        for attri in self.attributes:
            if attri in cate_atts:
                all_values = self.raw_data[attri].unique()
                feature_domain[attri] = (len(all_values), list(all_values))
            else:
                feature_domain[attri] = (min(self.raw_data[attri]), max(self.raw_data[attri]))
        self.metadata_desp.update({"domain": feature_domain})

        # remove the sensitive and target attributes from the inferred list since these attributes are treated separately in the pipeline
        if self.target_col in cate_atts:
            cate_atts.remove(self.target_col)
        if self.target_col in num_atts:
            num_atts.remove(self.target_col)
        for si in self.sensitive_atts:
            cate_atts.remove(si)

        self.num_atts = list(num_atts)
        self.cate_atts = list(cate_atts)

        log_dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.generate_file_path())
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)

        self.log_dir_name = log_dir_name
        print("**"*10, log_dir_name)
        # print(self.metadata_desp)

        with open(log_dir_name+"/"+self.data_name+'.json', 'w') as outfile:
            json.dump(dict(self.metadata_desp), outfile)

        return self

    def iter_steps(self, steps):
        # TODO: optimize with the input data.
        """
        Generator function to iterate steps.
        :param steps: list of objects that represent the steps user want to perform on the input data.
                      Supported steps are listed in STEPS.md.
        :return:  the pandas dataframes that are returned by applying a step on the input data.
        """
        # islice(steps, 0, len(steps))
        for idx, stepi in enumerate(steps):
            yield idx, stepi

    def validate_input_steps(self, steps):

        if not steps or not isinstance(steps, list):
            print("Require non-empty list of steps as input!")
            raise ValueError

        # TODO: integrity check for steps using pre-requisite order

        return self

    def fill_zero_to_dummy_data(self, input_step, df):
        raw_df = pd.get_dummies(self.raw_data, columns=self.cate_atts, prefix_sep='=')
        if input_step.step_name() in ["FairPreprocessor", "Model", "FairInprocessor", "FairPostprocessor"]: # these steps return non-encoded data
            encoded_df = pd.get_dummies(df, columns=self.cate_atts, prefix_sep='=')
        else:
            encoded_df = df
        # print("**" * 10, raw_df.shape, encoded_df.shape)
        if raw_df.shape[1] != encoded_df.shape[1]:
            for coli in set(raw_df.columns):
                if "=" in coli and coli[:coli.find("=")] in self.cate_atts and coli not in encoded_df.columns:
                    encoded_df.loc[:, coli] = 0

        # print("**" * 10, raw_df.shape, encoded_df.shape)
        return encoded_df

    def run_pipeline(self, steps, save_interdata=False):
        """

        :param df: pandas dataframe, the data on which steps are applied.
        :param steps: list of classes that represent the steps user want to perform on the above data.
                      Supported steps are listed in STEPS.md.
        :return:  two pandas dataframes: before_data and after_data. after_data is the data after applied the input steps.

        """
        self.validate_input_parameters()
        self.get_meta_information()

        self.validate_input_steps(steps)

        # first run the splitter
        print(self.print_log_message(steps, 0))
        train_df, val_df, test_df = steps[0].apply(self.raw_data)
        print(PRINT_SPLIT)

        # for step_idx, train_fitted_step, train_df in self.iter_steps(steps, train_df):
        # for step_idx, stepi in enumerate(steps):
        for step_idx, stepi in self.iter_steps(steps):
            if step_idx == 0:
                continue

            print(self.print_log_message(steps, step_idx))

            # run the same step on validation data, if multiple methods are specified at one step, the one with optimal performance on validation set is selected
            if isinstance(steps[step_idx], list): # multiple methods specified at one step
                # TODO: add the selection on validation data
                pass

            else: # single method specified at a step
                stepi.fit(train_df)
                train_df = stepi.apply(train_df)

                if stepi.fit_only_on_train(): # the steps that fit only on train
                    val_df = stepi.apply(val_df)
                    test_df = stepi.apply(test_df)

                else: # the steps that treat train, val, test independently through the same rule
                    stepi.fit(val_df)
                    val_df = stepi.apply(val_df)

                    stepi.fit(test_df)
                    test_df = stepi.apply(test_df)

            # autofill encoded columns after the steps that might affect the dimension of the encoded data. Only for categorical attributes that are not sensitive and target column
            if stepi.input_encoded_data() != stepi.output_encoded_data():
                train_df = self.fill_zero_to_dummy_data(stepi, train_df)
                val_df = self.fill_zero_to_dummy_data(stepi, val_df)
                test_df = self.fill_zero_to_dummy_data(stepi, test_df)

            if save_interdata: # save intermediate data on the disc
                for file_i, df_i in zip(["train", "val", "test"], [train_df, val_df, test_df]):
                    self.save_inter_data(file_i, df_i, steps[:step_idx+1])

            print(PRINT_SPLIT)

        # # TODO: move to other location
        # # transfer back to original values for encoded sensitive and target columns
        # for dfi in [train_df, val_df, test_df]:
        #     for atti in [self.target_col] + self.sensitive_atts:
        #         df_i[atti] = df_i[atti].apply(lambda x: self.sensi_target_value_mapping[atti][x])
        #
        #     if self.pred_target in df_i.columns:
        #         # TODO: check whether to keep the line for score prediction
        #         df_i[self.pred_target] = df_i[self.pred_target].apply(lambda x: int(x >= 0.5)) # for the model that returns a score instead of labels
        #         df_i[self.pred_target] = df_i[self.pred_target].apply(lambda x: self.sensi_target_value_mapping[self.target_col][x])

        return train_df, val_df, test_df

    def print_log_message(self, steps, step_idx):
        if not self.verbose:
            return None
        return '(step %d of %d) running %s' % (step_idx + 1, len(steps), steps[step_idx].abbr_name()) + PRINT_SPLIT

    def get_executed_steps_name(self, executed_steps):
        return "_".join([x.abbr_name() for x in executed_steps])

    def generate_file_path(self, file_name=''):
        dir_name = '{}__{}/'.format(self.exec_timestamp, self.data_name)
        return self.log_path + dir_name + file_name

    def generate_timestamp(self):
        return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]


    def save_inter_data(self, file_name, df: pd.DataFrame, executed_steps):
        data_file_name = '{}-{}.csv'.format(self.get_executed_steps_name(executed_steps), file_name)
        data_file_path = os.path.join(self.log_dir_name, data_file_name)

        df.to_csv(data_file_path, index=False)
        if self.verbose:
            print ("Data is saved to ", data_file_path)

    

if __name__ == '__main__':

    data_file = "../data/german_AIF_test.csv"
    y_col = "credit"
    y_posi = ["good"]
    sensi_atts = ["age", "sex"]

    value_mapping = {"female": 0, "male": 1, "good": 1, "bad": 0, "young": 0, "old": 1}

    debias_focus_att = "sex"
    global_seed = 0

    numerical_atts = ["month", "credit_amount"]
    categorical_atts = ["status", "employment", "housing"]

    pipeline = [RandomSplitter([0.5, 0.3, 0.2], global_seed), NoSampler(), NoImputer(), SK_MinMaxScaler(numerical_atts),
                  NoBinarizer(),  OneHotEncoder(categorical_atts), MappingEncoder([y_col] + sensi_atts, value_mapping),
                  AIF_Reweighing(y_col, debias_focus_att), OPT_LogisticRegression(y_col, global_seed), NoFairPostprocessor()]



    cur_pip = FairPipeline(data_file, y_col, y_posi, sensi_atts, value_mapping)


    train_now, val_now, test_now = cur_pip.run_pipeline(pipeline, save_interdata=True)

