"""
    Base abstract class for every step supported in this system.
"""
METHOD_NAME_MAPPING = {"RandomSplitter": "RSP", "BalanceTargetSplitter": "BTSP",
             "RandomSampler": "RSM", "BalancePopulationSampler": "BPSM",
             "DropNAImputer": "DNIM", "ModeImputer": "MIM", "DatawigImputer": "DWIM",
             "SK_StandardScaler": "SSC", "SK_MinMaxScaler": "MMSC",
             "SK_Discretizer": "DCA", "SK_Binarizer": "BCA",
             "SK_OrdinalEncoder": "OREN", "SK_OneHotEncoder": "OHEN",
             "CustomCateAttsEncoder": "CCSN",
             "AIF_Reweighing": "RWFB", "AIF_DIRemover": "DIRFB",
             "SK_LogisticRegression": "LRRM", "SK_DecisionTree": "DTRM", "OPT_LogisticRegression": "OLRRM", "OPT_DecisionTree": "ODTRM", "AIF_AdversarialDebiasing": "ADFM",
             "AIF_EqOddsPostprocessing": "EQFA", "AIF_CalibratedEqOddsPostprocessing": "CEQFA"}

SUPPORT_STEPS = {"SP": "Splitter", "SM": "Sampler", "IM": "Imputer", "SC": "Scaler",
                 "CA": "Categorizer", "EN": "Encoder", "SN": "SensitiveAttEncoder",
                 "FB": "FairPreprocessor", "RM": "model", "FM": "model", "RA": "FairPostprocessor"}

class Step():
    def fit(self, df):
        """
        :param df: pandas dataframe, stores the data to learn the step.
        :return: self: the fitted step is updated.
        """
        raise NotImplementedError

    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the step.
        :return: pandas dataframe, stores the data after the step.
        """
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def abbr_name(self):
        raise NotImplementedError

    def step_name(self):
        raise NotImplementedError

    def input_encoded_data(self):
        # use to detect the change of the dimension of the dataset
        raise NotImplementedError

    def output_encoded_data(self):
        # use to detect the change of the dimension of the dataset
        raise NotImplementedError

    def fit_only_on_train(self):
        # indicate whether the step fit on the input data (return False) or use the fitted model (return True)
        raise NotImplementedError


