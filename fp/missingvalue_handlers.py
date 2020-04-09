from sklearn.impute import SimpleImputer
import numpy as np
import datawig


class MissingValueHandler:

    def name(self):
        raise NotImplementedError

    def fit(self, df):
        raise NotImplementedError

    def handle_missing(self, df):
        raise NotImplementedError


class CompleteCaseAnalysis(MissingValueHandler):

    def name(self):
        return 'complete_case'

    def fit(self, data):
        pass

    def handle_missing(self, df):
        return df.dropna()


class ModeImputer(MissingValueHandler):

    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'mode_imputation'

    def fit(self, df):
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            self.imputers[column] = SimpleImputer(strategy='most_frequent').fit(values)

    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns:
            values = np.array(df[column]).reshape(-1, 1)
            imputed_df[column] = self.imputers[column].transform(values)
        return imputed_df


class DataWigSimpleImputer(MissingValueHandler):

    def __init__(self, columns_to_impute, label_column, out):
        self.columns_to_impute = columns_to_impute
        self.label_column = label_column
        self.out = out
        self.imputers = {}
        super().__init__()

    def name(self):
        return 'datawig_imputation'

    def fit(self, df):
        for column in self.columns_to_impute:
            input_columns = list(set(df.columns) - set(label_column, column))
            self.imputers[column] = datawig.SimpleImputer(input_columns=input_columns,
                                                          output_column=column, output_path=self.out).fit(train_df=df)
            
    def handle_missing(self, df):
        imputed_df = df.copy(deep=True)
        for column in self.columns_to_impute:
            imputed_df[column] = self.imputers[column].predict(df)[column + '_imputed']
        return imputed_df
