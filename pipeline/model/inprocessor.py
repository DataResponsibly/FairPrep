"""
    Super class for all the supported classifier classes including fair-classifiers.
"""
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pipeline.step import Step

class Model(Step):
    def __init__(self, step_name, step, df, target_col, instance_weights=[], hyper_tune=False, param_grid={}, sensitive_att=None, fair_aware=False, target_positive=1):
        """
        :param step_name: str, name of the current input step.
        :param step: object of the initialized class.
        :param df: pandas dataframe, stores the data.
        :param target_col: str, the name of the target attribute.
        :param instance_weights: list of float in [0,1], each float represents the weight of the sample in above data.
        :param hyper_tune: boolean, whether to tune the hyper-parameter. Default is False.
        :param param_grid: dict, stores the search range of the hyper-parameter. When hyper_tune is True, this must be provided.
        :param sensitive_att: str, the name of a sensitive attribute.
        :param fair_aware: boolean, whether the model is fair-aware. Default is False.
        :param target_positive: integer, 0 or 1, represents the positive value of the target attribute. Default is 1.
        """

        super().__init__(step_name=step_name, df=df, sensitive_att=sensitive_att, target_col=target_col)
        # assume the data set has been encoded to numerical values
        if fair_aware: # fair classifiers
            # intitialize a binary label dataset from AIF 360
            aif_df = BinaryLabelDataset(df=df, label_names=[target_col], protected_attribute_names=[sensitive_att])
            fitted_step = step.fit(aif_df)
            input_score = False
        else: # regular classifiers
            if len(instance_weights) == 0:
                instance_weights = [1 for _ in range(1, df.shape[0] + 1)]
            if hyper_tune: # grid search for best hyper parameters
                if not param_grid:
                    print("Need to specify the search range of the hyper parameters - 'param_grid' is empty!")
                    raise ValueError

                search = GridSearchCV(Pipeline([('learner', step)]), param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
                fitted_step = search.fit(np.array(df.drop(columns=[target_col])), np.array(df[target_col]), None, **{'learner__sample_weight': instance_weights})
            else:
                fitted_step = step.fit(np.array(df.drop(columns=[target_col])), np.array(df[target_col]), sample_weight=instance_weights)
            input_score = True

        self.input_score = input_score
        self.step = fitted_step
        self.target_positive = target_positive


    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned discretizer.
        :return: pandas dataframe, stores the data after discretize.
        """

        # initialize AIF360 BinaryLabelDataset

        if self.input_score:  # for regular model, generate score prediction
            aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[])
            after_df, _ = aif_pred_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)

            favorable_class_idx = list(self.step.classes_).index(self.target_positive)
            after_df[self.pred_target_col] = [x[favorable_class_idx] for x in self.step.predict_proba(np.array(df.drop(columns=[self.target_col])))]

        else:  # for fair model, generate label prediction
            aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col],
                                             protected_attribute_names=[self.sensitive_att])

            after_aif_df = self.step.predict(aif_pred_df)
            after_df, _ = after_aif_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
            after_df[self.pred_target_col] = after_aif_df.labels

        return after_df
