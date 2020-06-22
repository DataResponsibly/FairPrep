"""
    Super class for all the supported postprocessor classes.
"""
import numpy as np
from aif360.datasets import BinaryLabelDataset
from pipeline.step import Step

class Postprocessor(Step):
    def __init__(self, step_name, step, df, sensitive_att, target_col, input_score=True, clf_threshold=0.5):
        """
        :param step_name: str, name of the current input step.
        :param step: object of the initialized class.
        :param df: pandas dataframe, stores the data.
        :param sensitive_att: str, the name of a sensitive attribute.
        :param target_col: str, the name of the target attribute.
        :param input_score: boolean, represent whether the post-processor takes predicted score as input. Default is True.
        :param clf_threshold: float in [0, 1], represents the threshold to categorize class labels from predicted scores.
        """
        if "pred_"+target_col not in df.columns:
            print("Require the predictions for ",target_col, " existing in the data!")
            raise ValueError
        super().__init__(step_name=step_name, df=df, sensitive_att=sensitive_att, target_col=target_col)
        # assume the data set has been encoded to numerical values,
        # intitialize a BinaryLabelDataset from AIF 360
        aif_true_df = BinaryLabelDataset(df=df.drop(columns=["pred_"+target_col]), label_names=[target_col], protected_attribute_names=[sensitive_att])

        aif_pred_df = aif_true_df.copy()

        if input_score:
            aif_pred_df.scores = df["pred_"+target_col]
        else:
            aif_pred_df.labels = np.array([int(x >= clf_threshold) for x in df["pred_"+target_col]])
        self.input_score = input_score
        self.step = step.fit(aif_true_df, aif_pred_df)
        self.clf_threshold = clf_threshold



    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned discretizer.
        :return: pandas dataframe, stores the data after discretize.
        """

        # initialize AIF360 BinaryLabelDataset

        if self.input_score: # use score prediction to fit model, e.g. RejectOptionClassification, CalibratedEqOddsPostprocessing
            aif_pred_df = BinaryLabelDataset(df=df, label_names=[self.target_col], scores_names=[self.pred_target_col],
                                             protected_attribute_names=[self.sensitive_att])
        else: # use label prediction to fit model, e.g. EqOddsPostprocessing
            df["pred_label_"+self.target_col] = [int(x >= self.clf_threshold) for x in df[self.pred_target_col]]
            aif_pred_df = BinaryLabelDataset(df=df.drop(columns=[self.pred_target_col]), label_names=["pred_label_"+self.target_col],
                                         protected_attribute_names=[self.sensitive_att])

        after_aif_df = self.step.predict(aif_pred_df)
        after_df, _ = after_aif_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
        after_df[self.pred_target_col] = after_aif_df.labels

        return after_df