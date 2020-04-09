import pandas as pd


class TrainDataSampler:
    def name(self):
        raise NotImplementedError

    def sample(self, df):
        raise NotImplementedError


class CompleteData(TrainDataSampler):

    def name(self):
        return 'complete_data'

    def sample(self, df):
        return df


class BalancedExamplesSampler(TrainDataSampler):

    def __init__(self, label_name, positive_label, n, random_state):
        super().__init__()
        self.label_name = label_name
        self.positive_label = positive_label
        self.n = n
        self.random_state = random_state

    def name(self):
        return 'balanced_data_{}'.format(self.n)

    def sample(self, df):

        examples_per_class = int(self.n / 2)

        positive = df[df[self.label_name] == self.positive_label].sample(n=examples_per_class, random_state=self.random_state)
        negative = df[df[self.label_name] != self.positive_label].sample(n=examples_per_class, random_state=self.random_state)
        return pd.concat([positive, negative])
