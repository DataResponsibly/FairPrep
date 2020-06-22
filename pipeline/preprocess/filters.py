"""
    Classes to filter slice of data.
"""
import pandas as pd
from pipeline.preprocess.preprocessor import Preprocessor

# utility functions
def wrap_filter(att, value):
    if isinstance(value, str): # string or null
        if value in ["?", ""]: # for null
            return '{}!={}'.format(att, att)
        else:
            return '{}=="{}"'.format(att, value)
    else: # numerical value
        return '{}=={}'.format(att, value)

# TODO: add multiple filter class
class RowFilter(Preprocessor):

    def __init__(self, df, column, value):
        """
        :param column: str, name of the column name to be filtered.
        :param value: str, integer, float, the value of the column to be filtered.
        """
        if column is None or value is None:
            print("Need to specify column and value to create filter!")
            raise ValueError
        if column not in df.columns:
            print("Need to specify valid column!")
            raise ValueError
        if value not in df[column].unique():
            print("Need to specify valid value!")
            raise ValueError
        self.column = column
        self.value = value
        super().__init__(step_name="@".join(["RowFilter", column, str(value)]), df=df, focus_atts=[column], fit_flag=False)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to be filtered.
        :return: pandas dataframe, stores the data after filter.
        """
        return df.query(wrap_filter(self.column, self.value))

class RemoveColumnFilter(Preprocessor):
    def __init__(self, df, exclude_cols):
        """
        :param exclude_cols: list of string, each string represents the name of the column to be excluded.
        """

        super().__init__(step_name="RemoveColumnFilter", df=df, focus_atts=exclude_cols, fit_flag=False)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to be filtered.
        :return: pandas dataframe, stores the data after filter.
        """
        return df.drop(columns=self.focus_atts)

if __name__ == '__main__':

    data = pd.read_csv("../../data/adult_pre_reweigh.csv")
    # cur_o = RowFilter(data, "sex", 0)
    cur_o = RemoveColumnFilter(data, ["sex","race"])

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_" + cur_o.get_name() + ".csv", index=False)

    print(cur_o.get_name())
