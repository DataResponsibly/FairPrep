"""
    Classes to split data into train, validation, and test set.
"""
from pipeline.preprocess.preprocessor import Preprocessor
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def valid_split_ratio(input_ratios):
    if input_ratios is None:
        print("Need to specify split_ratio!")
        raise ValueError
    else:
        if len(input_ratios) == 1:  # not valid, at least two for train and test
            print("split_ratio should have at least 2 values for train and test sets and at most 3 values for train, validation and test sets!")
            raise ValueError
        if sum([not isinstance(x, float) for x in input_ratios]) > 0:
            print("split_ratio includes non float value!")
            raise ValueError
        for x in input_ratios:
            if not isinstance(x, float):
                print("split_ratio includes non float value!")
                raise ValueError
            else:
                if x < 0 or x > 1:
                    print("split_ratio includes not valid value! Value should between 0 and 1.")
                    raise ValueError
                if sum(input_ratios) != 1:
                    print("The sum of split_ratio does not equal to 1!")
                    raise ValueError

    return True

class BalanceTargetSplitter(Preprocessor):
    def __init__(self, df, split_ratio, target_col, seed=0):
        """
        :param df: pandas dataframe, stores the data to split.
        :param split_ratio: list of float, each float represents the size-ratio of splitted data. Corresponding order maps to the size of the train, [validataion], and test set.
                            Value ranges in [0,1]. Sum of the values in this list should be equal to 1.
                            e.g. [0.7, 0.2, 0.1] means 70% train, 20% validation, and 10% test set.
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, seed to be used to generate random state. Same as 'random_state' in sklearn.model_selection.StratifiedKFold. Default is 0.
        """
        super().__init__("@".join(["BalanceTargetSplitter", str(len(split_ratio))]), df=df, fit_flag=False, target_col=target_col)
        if valid_split_ratio(split_ratio):
            if len(split_ratio) == 2: # train and test
                train_size = split_ratio[0]
                test_size = split_ratio[1]
                self.splitters = [StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=seed)]
            else: # train, validation and test
                train_size = split_ratio[0]
                validation_size = split_ratio[1]
                test_size = split_ratio[2]

                self.splitters = [StratifiedShuffleSplit(n_splits=1, test_size=test_size+validation_size, train_size=train_size, random_state=seed),
                                  StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=validation_size, random_state=seed)]

        self.split_ratio = split_ratio
        self.seed = seed

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned splitter.
        :return: pandas dataframe, stores the data after split.
        """
        def split_to_df(splitter, input_df, y_label):
            after_df_1 = pd.DataFrame()
            after_df_2 = pd.DataFrame()
            X = np.array(df.drop(columns=[y_label]))
            y = np.array(df[y_label])
            for index_1, index_2 in splitter.split(X, y):
                X_1, X_2 = X[index_1], X[index_2]
                y_1, y_2 = y[index_1], y[index_2]

                after_df_1 = pd.concat([after_df_1, pd.DataFrame(data=np.hstack((X_1, y_1.reshape(-1,1))) ,columns=input_df.columns)])
                after_df_2 = pd.concat([after_df_2, pd.DataFrame(data=np.hstack((X_2, y_2.reshape(-1,1))), columns=input_df.columns)])
            return after_df_1, after_df_2

        if len(self.split_ratio) == 2: # without validation set
            return split_to_df(self.splitters[0], df, self.target_col)
        else: # with validation set
            after_train_df, rest_df = split_to_df(self.splitters[0], df, self.target_col)
            after_val_df, after_test_df = split_to_df(self.splitters[1], rest_df, self.target_col)
            return after_train_df, after_val_df, after_test_df

class RandomSplitter(Preprocessor):
    def __init__(self, df, split_ratio, seed=0):
        """
        :param df: pandas dataframe, stores the data to split.
        :param split_ratio: list of float, each float represents the size-ratio of splitted data. Corresponding order maps to the size of the train, [validataion], and test set.
                            Value ranges in [0,1]. Sum of the values in this list should be equal to 1.
                            e.g. [0.7, 0.2, 0.1] means 70% train, 20% validation, and 10% test set.
        :param seed: integer, seed to be used to generate random state. Same as 'random_state' in sklearn.model_selection.StratifiedKFold. Default is 0.
        """
        super().__init__("@".join(["RandomSplitter", str(len(split_ratio))]), df=df, fit_flag=False)

        if valid_split_ratio(split_ratio):
            self.split_ratio = split_ratio
        self.seed = seed

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned splitter.
        :return: pandas dataframe, stores the data after split.
        """
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        if len(self.split_ratio) == 2:  # without validation set
            split_idx = int(self.split_ratio[0]*df.shape[0])
            return df[:split_idx], df[split_idx:]
        else:  # with validation set
            split_idx_1 = int(self.split_ratio[0] * df.shape[0])
            split_idx_2 = split_idx_1 + int(self.split_ratio[1] * df.shape[0])

            return df[:split_idx_1], df[split_idx_1:split_idx_2], df[split_idx_2:]


if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_RandomSampler_1000.csv")
    # cur_o = BalanceTargetSplitter(data, [0.7, 0.3], "income-per-year")
    cur_o = RandomSplitter(data, [0.5, 0.3, 0.2])

    after_train, after_val, after_test = cur_o.apply(data)
    # after_train, after_test = cur_o.apply(data)
    print(after_train.shape)
    print(after_val.shape)
    print(after_test.shape)
    # after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())





