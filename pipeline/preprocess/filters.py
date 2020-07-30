"""
    Classes to filter slice of data.
"""
import pandas as pd
from pipeline.step import Step


# TODO: add multiple filter class
class RowFilter(Step):

    def __init__(self, column, value):
        """
        :param column: str, name of the column name to be focused.
        :param value: str, integer, float, the value of the column to keep.
        """

        self.column = column
        self.value = value

    def fit(self, df):
        pass

    # utility functions to wrap the input filter to a df query
    def wrap_filter(self):
        if isinstance(self.value, str):  # string or null
            if self.value in ["?", ""]:  # for null
                return '{}!={}'.format(self.column, self.column)
            else:
                return '{}=="{}"'.format(self.column, self.value)
        else:  # numerical value
            return '{}=={}'.format(self.column, self.value)

    def apply(self, df):
        return df.query(self.wrap_filter())

    def name(self):
        return "RowFilter"

    def abbr_name(self):
        return "RF"

    def step_name(self):
        return "Filter"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

class ColumnFilter(Step):
    def __init__(self, remove_cols):
        """
        :param exclude_cols: list of string, each string represents the name of the column to be filtered out.
        """
        self.remove_cols = remove_cols

    def fit(self, df):
        pass

    def apply(self, df):
        return df.drop(columns=self.remove_cols)

    def name(self):
        return "ColumnFilter"

    def abbr_name(self):
        return "CF"

    def step_name(self):
        return "Filter"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

if __name__ == '__main__':

    data = pd.read_csv("../../data/german_AIF.csv")
    cur_o = RowFilter("sex", "female")
    # cur_o = ColumnFilter(["sex", "age"])

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_AIF_" + cur_o.name() + ".csv", index=False)

