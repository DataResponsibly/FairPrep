"""
    Super class for all the supported preprocessor classes.
"""
import numpy as np
from aif360.datasets import BinaryLabelDataset
from pipeline.step import Step

class Preprocessor(Step):
    def __init__(self, step_name, df, step=None, focus_atts=[], fit_flag=True, weight_flag=False, sensitive_att=None, target_col=None, fair_aware=False, na_mark=None):
        """
        :param step_name: str, name of the current input step.
        :param df: pandas dataframe, stores the data.
        :param step: object of the initialized class. If none, initialize here.
        :param focus_atts: lisf of str, each str represents the name of a column in above data that will be pre-processed.
        :param fit_flag: boolean, whether to initialize step object here.
        :param weight_flag: boolean, whether to output extra sample weight after fair-preprocessor.
        :param sensitive_att: str, the name of a sensitive attribute.
        :param target_col: str, the name of the target attribute.
        :param fair_aware: boolean, whether the preprocessor is fair-aware. Default is False. If true, sensitive_att and target_col can not be null.
        """
        super().__init__(step_name=step_name, df=df, focus_atts=focus_atts, sensitive_att=sensitive_att, target_col=target_col)

        if len(focus_atts) > 0 and fit_flag:
            fitted_step = {}
            for idx, ai in enumerate(focus_atts):
                fitted_step[ai] = step[ai].fit(np.array(df[ai]).reshape(-1, 1))
            self.step = fitted_step
        elif fair_aware and fit_flag: # for fair-preprocessors
            aif_df = BinaryLabelDataset(df=df, label_names=[target_col], protected_attribute_names=[sensitive_att])
            self.step = step.fit(aif_df)
        else:
            if step is not None:
                self.step = step

        # address different encoding of missing values
        if na_mark is not None:
            self.na_mark = na_mark
        else:
            self.na_mark = None
        self.fair_aware = fair_aware
        self.fit_flag = fit_flag
        self.weight_flag = weight_flag


    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned discretizer.
        :return: pandas dataframe, stores the data after discretize.
        """
        if self.na_mark:
            df = df.replace({self.na_mark:np.nan})
        if self.fair_aware: # fair-preprocessor
            aif_df = BinaryLabelDataset(df=df, label_names=[self.target_col], protected_attribute_names=[self.sensitive_att])
            if self.fit_flag: # fit has been initialized
                after_aif_df = self.step.transform(aif_df)
            else: # fit and transform is combined, e.g. DisparateImpactRemover
                after_aif_df = self.step.fit_transform(aif_df)

            after_df, _ = after_aif_df.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
            if self.weight_flag:
                preprocessed_weights = after_aif_df.instance_weights

        else: # regular preprocessor
            after_df = df.copy()
            for ai in self.focus_atts:
                after_df[ai] = self.step[ai].transform(np.array(after_df[ai]).reshape(-1, 1))

        if self.weight_flag: # for the preprocessor that updates weights, e.g. Reweighing
            return after_df, preprocessed_weights
        else:
            return after_df