"""
    Classes of supervised binary classifiers.
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pipeline.step import Step


class SK_LogisticRegression(Step):
    def __init__(self, target_col, seed, loss_func="log", instance_weights=[], target_positive=1):
        """
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, the seed for random state.
        :param loss_func: str, the name of the loss function used in linear model. Same as the loss parameter in sklearn.linear_model.SGDClassifier.
                         The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param target_positive: integer, 0 or 1, represents the positive value of the target attribute. Default is 1.

        """
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.loss_func = loss_func
        self.instance_weights = instance_weights
        self.target_positive = target_positive
        self.fitted_step = None

    def fit(self, df):
        if len(self.instance_weights) == 0:
            self.instance_weights = [1 for _ in range(1, df.shape[0] + 1)]

        self.fitted_step = SGDClassifier(loss=self.loss_func, random_state=self.seed).fit(np.array(df.drop(columns=[self.target_col])), np.array(df[self.target_col]), sample_weight=self.instance_weights)

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[])
        after_df, _ = aif_pred_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

        favorable_class_idx = list(self.fitted_step.classes_).index(self.target_positive)
        after_df[self.pred_target_col] = [x[favorable_class_idx] for x in self.fitted_step.predict_proba(np.array(df.drop(columns=[self.target_col])))]

        return after_df

    def name(self):
        return "SK_LogisticRegression"

    def abbr_name(self):
        return "LR"

    def step_name(self):
        return "Model"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True


class SK_DecisionTree(Step):
    def __init__(self, target_col, seed, instance_weights=[], target_positive=1):
        """
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, the seed for random state.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param target_positive: integer, 0 or 1, represents the positive value of the target attribute. Default is 1.

        """

        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.instance_weights = instance_weights
        self.target_positive = target_positive
        self.fitted_step = None

    def fit(self, df):
        if len(self.instance_weights) == 0:
            self.instance_weights = [1 for _ in range(1, df.shape[0] + 1)]

        self.fitted_step = DecisionTreeClassifier(random_state=self.seed).fit(np.array(df.drop(columns=[self.target_col])), np.array(df[self.target_col]), sample_weight=self.instance_weights)

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[])
        after_df, _ = aif_pred_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

        favorable_class_idx = list(self.fitted_step.classes_).index(self.target_positive)
        after_df[self.pred_target_col] = [x[favorable_class_idx] for x in self.fitted_step.predict_proba(np.array(df.drop(columns=[self.target_col])))]

        return after_df

    def name(self):
        return "SK_DecisionTree"

    def abbr_name(self):
        return "DT"

    def step_name(self):
        return "Model"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class OPT_LogisticRegression(Step):
    def __init__(self, target_col, seed, loss_func="log", max_iter=1000, instance_weights=[], target_positive=1):
        """
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, random seed.
        :param loss_func: str, the name of the loss function used in linear model. Same as the loss parameter in sklearn.linear_model.SGDClassifier.
                         The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        :param max_iter: integer, max number of iterations of the model.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param target_positive: integer, 0 or 1, represents the positive value of the target attribute. Default is 1.
        """
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.loss_func = loss_func
        self.max_iter = max_iter
        self.instance_weights = instance_weights
        self.target_positive = target_positive
        self.fitted_step = None

        self.param_grid = {
            'learner__loss': [loss_func],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.00005, 0.0001, 0.005, 0.001]
        }

    def fit(self, df):
        if len(self.instance_weights) == 0:
            self.instance_weights = [1 for _ in range(1, df.shape[0] + 1)]

        search = GridSearchCV(Pipeline([('learner', SGDClassifier(max_iter=self.max_iter, random_state=self.seed))]), self.param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        self.fitted_step = search.fit(np.array(df.drop(columns=[self.target_col])), np.array(df[self.target_col]), None,
                                 **{'learner__sample_weight': self.instance_weights})

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[])
        after_df, _ = aif_pred_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

        favorable_class_idx = list(self.fitted_step.classes_).index(self.target_positive)
        after_df[self.pred_target_col] = [x[favorable_class_idx] for x in self.fitted_step.predict_proba(np.array(df.drop(columns=[self.target_col])))]

        return after_df

    def name(self):
        return "OPT_LogisticRegression"

    def abbr_name(self):
        return "OLR"

    def step_name(self):
        return "Model"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

class OPT_DecisionTree(Step):
    def __init__(self, target_col, seed, instance_weights=[], target_positive=1):
        """
        :param target_col: str, the name of the target variable in above data.
        :param seed: integer, random seed.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param target_positive: integer, 0 or 1, represents the positive value of the target attribute. Default is 1.
        """
        self.target_col = target_col
        self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        self.seed = seed
        self.instance_weights = instance_weights
        self.target_positive = target_positive
        self.fitted_step = None

        self.param_grid = {
            'learner__min_samples_split': range(20, 500, 10),
            'learner__max_depth': range(15, 30, 2),
            'learner__min_samples_leaf': [3, 4, 5, 10],
            "learner__criterion": ["gini", "entropy"]
        }

    def fit(self, df):
        if len(self.instance_weights) == 0:
            self.instance_weights = [1 for _ in range(1, df.shape[0] + 1)]

        search = GridSearchCV(Pipeline([('learner', DecisionTreeClassifier(random_state=self.seed))]), self.param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        self.fitted_step = search.fit(np.array(df.drop(columns=[self.target_col])), np.array(df[self.target_col]), None,
                                 **{'learner__sample_weight': self.instance_weights})

        return self

    def apply(self, df):
        aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[])
        after_df, _ = aif_pred_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

        favorable_class_idx = list(self.fitted_step.classes_).index(self.target_positive)
        after_df[self.pred_target_col] = [x[favorable_class_idx] for x in self.fitted_step.predict_proba(np.array(df.drop(columns=[self.target_col])))]

        return after_df

    def name(self):
        return "OPT_DecisionTree"

    def abbr_name(self):
        return "ODT"

    def step_name(self):
        return "Model"

    def input_encoded_data(self):
        return True

    def output_encoded_data(self):
        return False

    def fit_only_on_train(self):
        return True

if __name__ == '__main__':
    data = pd.read_csv("../../data/german_pre_encoded.csv")
    # cur_o = SK_LogisticRegression("credit", 0)
    cur_o = SK_DecisionTree("credit", 0)
    # cur_o = OPT_LogisticRegression("credit", 0)
    # cur_o = OPT_DecisionTree("credit", 0)

    cur_o.fit(data)
    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/german_after_" + cur_o.name() + ".csv", index=False)
