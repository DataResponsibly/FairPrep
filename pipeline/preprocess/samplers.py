"""
    Classes to sample a subset of data.
"""

import numpy as np
import pandas as pd
from pipeline.preprocess.preprocessor import Preprocessor

class RandomSampler(Preprocessor):
    def __init__(self, df, sample_n, random_state=0):
        """
        :param sample_n: integer, the size of the sampled subset of data
        """
        if not sample_n:
            print("Need to specify a size greater than 0!")
            raise ValueError
        self.sample_n = sample_n
        self.random_state = random_state
        super().__init__("RandomSampler@"+str(sample_n), df=df, fit_flag=False)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to be sampled.
        :return: pandas dataframe, stores the data after sample.
        """
        return df.sample(n=self.sample_n, random_state=self.random_state)

class BalancePopulationSampler(Preprocessor):
    def __init__(self, df, sample_n, balance_col, random_state=0):
        """
        :param sample_n: integer, the size of the sampled subset of data
        :param balance_col: str, the name of a categorical column that the population of groups within this column are balanced in the sampled subset.
        :param random_state: integer, the seed for random process, same as random_state in pandas.DataFrame.sample.

        """
        if not sample_n:
            print("Need to specify a size greater than 0!")
            raise ValueError
        if not balance_col:
            print("Need to specify the name of a column to perform balance sampling within this column!")
            raise ValueError
        if balance_col not in df.columns:
            print("Need to specify a valid column to perform balance sampling within this column!")
            raise ValueError
        self.sample_n = sample_n
        self.balance_col = balance_col
        self.random_state = random_state
        super().__init__("@".join(["BalanceSampler", balance_col, str(sample_n)]), df=df, fit_flag=False)

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to be sampled.
        :return: pandas dataframe, stores the data after sample.
        """
        # TODO: update to minimum sample set
        balance_groups = list(df[self.balance_col].unique())
        n_group = int(np.ceil(self.sample_n/len(balance_groups)))
        # print(n_group)
        sampled_df = {}
        small_groups = []
        for gi in balance_groups:
            gi_data = df[df[self.balance_col]==gi]
            if gi_data.shape[0] < n_group: # for small group, accept all in the balanced samples
                sampled_df[gi] = gi_data
                small_groups.append(gi)
            else:
                sampled_df[gi] = df[df[self.balance_col]==gi].sample(n=n_group, random_state=self.random_state)

        after_df = pd.DataFrame()
        if not self.sample_n % len(balance_groups): # for even groups
            for gi in balance_groups:
                after_df = pd.concat([after_df, sampled_df[gi]])
        else: # for odd groups, remove extra sampled items
            remove_cates = list(set(balance_groups).difference(small_groups))
            print(len(balance_groups)*n_group-self.sample_n)
            for gi in np.random.choice(remove_cates, len(balance_groups)*n_group-self.sample_n, False):
                after_df = pd.concat([after_df, sampled_df[gi].head(sampled_df[gi].shape[0]-1)])
                print(gi, after_df.shape[0])
                del sampled_df[gi] # remove the group already added into the final sample
            print()
            print(sampled_df.keys())
            for gi in sampled_df:
                after_df = pd.concat([after_df, sampled_df[gi]])
                print(gi, after_df.shape[0])
        print()
        print(after_df.groupby(self.balance_col).count())
        return after_df

if __name__ == '__main__':
    # cur_o = BalancePopulationSampler(1000, "marital-status")
    data = pd.read_csv("../../data/adult.csv")
    cur_o = RandomSampler(data, 1000)

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_" + cur_o.get_name() + ".csv", index=False)
    print(cur_o.get_name())