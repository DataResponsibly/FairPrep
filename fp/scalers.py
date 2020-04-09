from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class NoScaler(BaseEstimator, TransformerMixin):

    def name(self):
        return 'no_scaler'

    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array


class NamedStandardScaler(StandardScaler):

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy, with_mean, with_std)

    def name(self):
        return 'standard_scaler'


class NamedMinMaxScaler(MinMaxScaler):

    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range, copy)

    def name(self):
        return 'minmax_scaler'
