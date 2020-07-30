"""
    Classes to split data into train, validation, and test set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pipeline.step import Step

# TODO: optimize the below function
def split_to_df(splitter, input_df, y_label):
    after_df_1 = pd.DataFrame()
    after_df_2 = pd.DataFrame()
    X = np.array(input_df.drop(columns=[y_label]))
    y = np.array(input_df[y_label])
    for index_1, index_2 in splitter.split(X, y):
        X_1, X_2 = X[index_1], X[index_2]
        y_1, y_2 = y[index_1], y[index_2]

        after_df_1 = pd.concat(
            [after_df_1, pd.DataFrame(data=np.hstack((X_1, y_1.reshape(-1, 1))), columns=input_df.columns)])
        after_df_2 = pd.concat(
            [after_df_2, pd.DataFrame(data=np.hstack((X_2, y_2.reshape(-1, 1))), columns=input_df.columns)])
    return after_df_1, after_df_2

def valid_split_ratio(split_ratio):
    if split_ratio is None:
        print("Need to specify split_ratio!")
        raise ValueError
    else:
        if len(split_ratio) == 1:  # not valid, at least two for train and test
            print(
                "split_ratio should have at least 2 values for train and test sets and at most 3 values for train, validation and test sets!")
            raise ValueError
        if sum([not isinstance(x, float) for x in split_ratio]) > 0:
            print("split_ratio includes non float value!")
            raise ValueError
        for x in split_ratio:
            if not isinstance(x, float):
                print("split_ratio includes non float value!")
                raise ValueError
            else:
                if x < 0 or x > 1:
                    print("split_ratio includes not valid value! Value should between 0 and 1.")
                    raise ValueError
                if sum(split_ratio) != 1:
                    print("The sum of split_ratio does not equal to 1!")
                    raise ValueError

    return True

class BalanceTargetSplitter(Step):
    # TODO: fix the bug of not returning enough item
    def __init__(self, split_ratio, target_col, seed):
        """
        :param split_ratio: list of float, each float represents the size-ratio of splitted data. Corresponding order maps to the size of the train, [validataion], and test set.
                            Value ranges in [0,1]. Sum of the values in this list should be equal to 1.
                            e.g. [0.7, 0.2, 0.1] means 70% train, 20% validation, and 10% test set.
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, seed to be used to generate random state. Same as 'random_state' in sklearn.model_selection.StratifiedKFold. Default is 0.
        """
        self.split_ratio = split_ratio
        self.target_col = target_col
        self.seed = seed

        self.fitted_step = None


    def fit(self, df):
        if valid_split_ratio(self.split_ratio):
            train_size = self.split_ratio[0]
            validation_size = self.split_ratio[1]
            test_size = self.split_ratio[2]

            self.fitted_step = [StratifiedShuffleSplit(n_splits=1, test_size=test_size+validation_size, train_size=train_size, random_state=self.seed),
                              StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=validation_size, random_state=self.seed)]
        else:
            print("Invalid inputs!")
            raise ValueError
        return self

    def apply(self, df):
        after_train_df, rest_df = split_to_df(self.fitted_step[0], df, self.target_col)
        after_val_df, after_test_df = split_to_df(self.fitted_step[1], rest_df, self.target_col)
        return after_train_df, after_val_df, after_test_df

    def name(self):
        return "BalanceTargetSplitter"

    def abbr_name(self):
        return "BS"

    def step_name(self):
        return "Splitter"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class RandomSplitter(Step):
    def __init__(self, split_ratio, seed):
        """
        :param split_ratio: list of float, each float represents the size-ratio of splitted data. Corresponding order maps to the size of the train, [validataion], and test set.
                            Value ranges in [0,1]. Sum of the values in this list should be equal to 1.
                            e.g. [0.7, 0.2, 0.1] means 70% train, 20% validation, and 10% test set.
        :param seed: integer, seed to be used to generate random state. Same as 'random_state' in sklearn.model_selection.StratifiedKFold. Default is 0.
        """
        self.split_ratio = split_ratio
        self.seed = seed
        self.fitted_step = None

    def fit(self, df):
        if valid_split_ratio(self.split_ratio):
            pass
        else:
            print("Invalid inputs!")
            raise ValueError

    def apply(self, df):
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        split_idx_1 = int(self.split_ratio[0] * df.shape[0])
        split_idx_2 = split_idx_1 + int(self.split_ratio[1] * df.shape[0])

        return df.iloc[:split_idx_1], df.iloc[split_idx_1:split_idx_2], df.iloc[split_idx_2:]

    def name(self):
        return "RandomSplitter"

    def abbr_name(self):
        return "RS"

    def step_name(self):
        return "Splitter"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_AIF.csv")
    print(data.shape)
    # cur_o = BalanceTargetSplitter([0.5, 0.3, 0.2], "credit", 0) # bug: returned splitted data's total size does not equal to the input data
    cur_o = RandomSplitter([0.5, 0.3, 0.2], 0)

    cur_o.fit(data)
    after_train, after_val, after_test = cur_o.apply(data)
    # after_train, after_test = cur_o.apply(data)
    print(after_train.shape)
    print(after_val.shape)
    print(after_test.shape)

    after_train.to_csv("../../data/german_AIF_train_"+cur_o.name()+".csv", index=False)
    after_val.to_csv("../../data/german_AIF_val_"+cur_o.name()+".csv", index=False)
    after_test.to_csv("../../data/german_AIF_test_"+cur_o.name()+".csv", index=False)





