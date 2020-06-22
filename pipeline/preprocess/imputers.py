"""
    Classes to impute missing values in data.
"""
import numpy as np
import pandas as pd
import datawig
from pipeline.preprocess.preprocessor import Preprocessor
from sklearn.impute import SimpleImputer

class DropNAImputer(Preprocessor):
    def __init__(self, df, na_mark=None):
        """
        :param df: pandas dataframe, stores the data to fit the imputer.
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        """
        super().__init__("DropNAImputer", df=df, fit_flag=False, na_mark=na_mark)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to impute.
        :return: pandas dataframe, stores the data after impute.
        """
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        return df.dropna()

class ModeImputer(Preprocessor):
    def __init__(self, df, num_atts, cate_atts, na_mark=None):
        """
        :param df: pandas dataframe, stores the data to fit the imputer.
        :param num_atts: list of str, each str represents the name of numerical column to be imputed using the mean value.
        :param cate_atts: list of str, each str represents the name of categorical column to be imputed using the most frequent value.
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        """
        if len(set(num_atts).intersection(cate_atts)) > 0:
            print("Some attributes are both in num_atts and cate_atts!")
            raise ValueError

        cur_step = {}
        if len(cate_atts) > 0:
            for ci in cate_atts:
                cur_step[ci] = SimpleImputer(strategy='most_frequent')
        if len(num_atts) > 0:
            for ni in num_atts:
                cur_step[ni] = SimpleImputer(strategy='mean')

        super().__init__("@".join(["ModeImputer"]+num_atts+cate_atts), df, step=cur_step, focus_atts=cate_atts+num_atts, fit_flag=True, na_mark=na_mark)


class DatawigImputer(Preprocessor):
    def __init__(self, df, impute_atts, na_mark=None, output_path="datawig/", num_epochs=50):
        """
        :param df: pandas dataframe, stores the data to fit the imputer.
        :param impute_atts: list of str, each str represents the name of column to be imputed using datawig model. Column can be categorical or numerical.
        :param na_mark: str, represents the symbol of missing values. Default is None, i.e. NaN represents the missing values.
        :param output_path: str, the path to store the learned datawig model.
        :param num_epochs: integer, the maximum iteration of datawig model.
        """
        super().__init__("@".join(["DatawigImputer"] + impute_atts), df, focus_atts=impute_atts, fit_flag=False, na_mark=na_mark)

        learned_imputers = {}
        for ai in impute_atts:
            learned_imputers[ai] = datawig.SimpleImputer(input_columns=list(set(df.columns).difference(ai)),
                                                          output_column=ai, output_path=output_path).fit(train_df=df, num_epochs=num_epochs)
        self.step = learned_imputers

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned imputer.
        :return: pandas dataframe, stores the data after impute.
        """
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        after_df = df.copy()
        for ai in self.focus_atts:
            after_df[ai] = self.step[ai].predict(df)[ai + '_imputed']
        return after_df

if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_RandomSampler_1000.csv")
    cur_o = DropNAImputer(data, na_mark="?")
    # cur_o = ModeImputer(data, ["fnlwgt"], ["workclass"], na_mark="?")
    # cur_o = DatawigImputer(data, ["workclass"], na_mark="?")

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())