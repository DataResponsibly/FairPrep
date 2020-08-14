"""
    Classes to impute missing values in data.
"""
import numpy as np
import pandas as pd
import datawig
from sklearn.impute import SimpleImputer
from FairPrep.step import Step

class NoImputer(Step):
    def __init__(self):
        self.fitted_step = None

    def fit(self, df):
        pass

    def apply(self, df):
        return df

    def name(self):
        return "NoImputer"

    def abbr_name(self):
        return "NI"

    def step_name(self):
        return "Imputer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

class DropNAImputer(Step):
    def __init__(self, na_mark=None):
        """
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        """
        self.na_mark = na_mark

    def fit(self, df):
        pass

    def apply(self, df):
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        return df.dropna()

    def name(self):
        return "DropNAImputer"

    def abbr_name(self):
        return "DN"

    def step_name(self):
        return "Imputer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

class ModeImputer(Step):
    def __init__(self, num_atts, cate_atts, na_mark=None):
        """
        :param num_atts: list of str, each str represents the name of numerical column to be imputed using the mean value.
        :param cate_atts: list of str, each str represents the name of categorical column to be imputed using the most frequent value.
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        """
        self.num_atts = num_atts
        self.cate_atts = cate_atts
        self.na_mark = na_mark

        self.fitted_step = None

    def fit(self, df):
        fitted_step = {}
        if len(self.cate_atts) > 0:
            for ci in self.cate_atts:
                fitted_step[ci] = SimpleImputer(strategy='most_frequent').fit(np.array(df[ci]).reshape(-1, 1))
        if len(self.num_atts) > 0:
            for ni in self.num_atts:
                fitted_step[ni] = SimpleImputer(strategy='mean').fit(np.array(df[ni]).reshape(-1, 1))
        self.fitted_step = fitted_step

        return self

    def apply(self, df):
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        after_df = df.copy()
        for ai in self.num_atts + self.cate_atts:
            after_df[ai] = self.fitted_step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        return after_df

    def name(self):
        return "SK_ModeImputer"

    def abbr_name(self):
        return "MI"

    def step_name(self):
        return "Imputer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class DatawigImputer(Step):
    def __init__(self, impute_atts, na_mark=None, output_path="datawig/", num_epochs=50):
        """
        :param impute_atts: list of str, each str represents the name of column to be imputed using datawig model. Column can be categorical or numerical.
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        :param output_path: str, the path to store the learned datawig model.
        :param num_epochs: integer, the maximum iteration of datawig model.
        """
        self.focus_atts = impute_atts
        self.na_mark = na_mark
        self.output_path = output_path
        self.num_epochs = num_epochs

        self.fitted_step = None

    def fit(self, df):
        learned_imputers = {}
        for ai in self.focus_atts:
            learned_imputers[ai] = datawig.SimpleImputer(input_columns=list(set(df.columns).difference(ai)), output_column=ai, output_path=self.output_path).fit(train_df=df, num_epochs=self.num_epochs)
        self.fitted_step = learned_imputers
        return self

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned imputer.
        :return: pandas dataframe, stores the data after impute.
        """
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.fitted_step[ai].predict(df)[ai + '_imputed']
        return after_df

    def name(self):
        return "DatawigImputer"

    def abbr_name(self):
        return "DW"

    def step_name(self):
        return "Imputer"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_AIF.csv")
    # cur_o = DropNAImputer(na_mark="?")
    cur_o = ModeImputer([],["workclass"], na_mark="?")
    # cur_o = DatawigImputer(["workclass"], na_mark="?") # TODO: test after fix dependency issue

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_AIF_"+cur_o.name()+".csv", index=False)
