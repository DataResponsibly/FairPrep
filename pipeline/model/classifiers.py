"""
    Classes of supervised binary classifiers.
"""

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from pipeline.model.inprocessor import Model


class SK_LogisticRegression(Model):
    def __init__(self, df, target_col, loss_func="log", instance_weights=[], seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the classifier.
        :param target_col: str, the name of the target variable in above data.
        :param loss_func: str, the name of the loss function used in linear model. Same as the loss parameter in sklearn.linear_model.SGDClassifier.
                         The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param seed: integer, the seed for random state.
        """

        cur_step = SGDClassifier(loss=loss_func, random_state=seed)
        super().__init__("@".join(["SK_LogisticRegression", target_col]), cur_step, df, target_col, instance_weights=instance_weights)


class SK_DecisionTree(Model):
    def __init__(self, df, target_col, instance_weights=[], seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the classifier.
        :param target_col: str, the name of the target variable in above data.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param seed: integer, the seed for random state.
        """
        cur_step = DecisionTreeClassifier(random_state=seed)
        super().__init__("@".join(["SK_DecisionTree", target_col]), cur_step, df, target_col, instance_weights=instance_weights)


class OPT_LogisticRegression(Model):
    def __init__(self, df, target_col, loss_func="log", max_iter=1000, instance_weights=[], seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the classifier.
        :param target_col: str, the name of the target variable in above data.
        :param loss_func: str, the name of the loss function used in linear model. Same as the loss parameter in sklearn.linear_model.SGDClassifier.
                         The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        :param max_iter: integer, max number of iterations of the model.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param seed: integer, random seed.
        """
        # Update below parameters according to the loss function used
        param_grid = {
            'learner__loss': [loss_func],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.00005, 0.0001, 0.005, 0.001]
        }
        cur_step = SGDClassifier(max_iter=max_iter, random_state=seed)
        super().__init__("@".join(["OPT_LogisticRegression", target_col]), cur_step, df, target_col, instance_weights=instance_weights, hyper_tune=True, param_grid=param_grid)

class OPT_DecisionTree(Model):
    def __init__(self, df, target_col, instance_weights=[], seed=0):
        """
        :param df: pandas dataframe, stores the data to fit the classifier.
        :param target_col: str, the name of the target variable in above data.
        :param instance_weights: list of float, each number represents the weight of the sample in above data.
        :param seed: integer, random seed.
        """
        param_grid = {
            'learner__min_samples_split': range(20, 500, 10),
            'learner__max_depth': range(15, 30, 2),
            'learner__min_samples_leaf': [3, 4, 5, 10],
            "learner__criterion": ["gini", "entropy"]
        }

        cur_step = DecisionTreeClassifier(random_state=seed)
        super().__init__("@".join(["OPT_DecisionTree", target_col]), cur_step, df, target_col, instance_weights=instance_weights, hyper_tune=True, param_grid=param_grid)


if __name__ == '__main__':
    data = pd.read_csv("../../data/adult_pre_reweigh.csv")
    cur_o = SK_LogisticRegression(data, "income-per-year")
    # cur_o = SK_DecisionTree(data, "income-per-year")
    # cur_o = OPT_LogisticRegression(data, "income-per-year")
    # cur_o = OPT_DecisionTree(data, "income-per-year")

    after_data = cur_o.apply(data)
    after_data.to_csv("../../data/adult_"+cur_o.get_name()+".csv", index=False)

    print(cur_o.get_name())