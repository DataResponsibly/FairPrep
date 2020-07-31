"""
    Classes to transform data for multiple columns.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from pipeline.preprocess.preprocessor import Preprocessor

class SK_ColumnTransformer(Preprocessor):
    def __init__(self, df, cols):
        """
        :param df: pandas dataframe, stores the data to fit the discretizer.
        :param cols: list of str, each str represents the name of an attribute in above data to be transformed.
        """
        pass
    # TODO: add column transformer

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned discretizer.
        :return: pandas dataframe, stores the data after discretize.
        """
        pass