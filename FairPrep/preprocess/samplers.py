"""
    Classes to sample a subset of data.
"""

import numpy as np
import pandas as pd
from FairPrep.step import Step


class NoSampler(Step):
    def __init__(self):
        self.fitted_step = None

    def fit(self, df):
        pass

    def apply(self, df):
        return df

    def name(self):
        return "NoSampler"

    def abbr_name(self):
        return "NS"

    def step_name(self):
        return "Sampler"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

class RandomSampler(Step):
    def __init__(self, sample_n, seed):
        """
        :param sample_n: integer, the size of the sampled subset of data
        :param seed: integer, the seed for random process.
        """
        self.sample_n = sample_n
        self.seed = seed

    def fit(self, df):
        pass

    def apply(self, df):
        return df.sample(n=self.sample_n, random_state=self.seed)

    def name(self):
        return "RandomSampler"

    def abbr_name(self):
        return "RS"

    def step_name(self):
        return "Sampler"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False


class BalancePopulationSampler(Step):
    def __init__(self, sample_n, balance_col, seed):
        """
        :param sample_n: integer, the size of the sampled subset of data
        :param balance_col: str, the name of a categorical column that the population of groups within this column are balanced in the sampled subset.
        :param seed: integer, the seed for random process.

        """

        self.sample_n = sample_n
        self.balance_col = balance_col
        self.seed = seed

    def fit(self, df):
        pass

    def apply(self, df):
        # TODO: update to minimum sample set and remove print statement
        balance_groups = list(df[self.balance_col].unique())
        n_group = int(np.ceil(self.sample_n/len(balance_groups)))

        sampled_df = {}
        small_groups = []
        for gi in balance_groups:
            gi_data = df[df[self.balance_col]==gi]
            if gi_data.shape[0] < n_group: # for small group, accept all in the balanced samples
                sampled_df[gi] = gi_data
                small_groups.append(gi)
            else:
                sampled_df[gi] = df[df[self.balance_col]==gi].sample(n=n_group, random_state=self.seed)

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

    def name(self):
        return "BalanceSampler"

    def abbr_name(self):
        return "BS"

    def step_name(self):
        return "Sampler"

    def input_encoded_data(self):
        return False

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return False

if __name__ == '__main__':

    data = pd.read_csv("../../data/german_AIF.csv")
    # cur_o = RandomSampler(200, 0)

    cur_o = BalancePopulationSampler(200, "sex", 0)

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_AIF_" + cur_o.name() + ".csv", index=False)