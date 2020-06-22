"""
    Base abstract class for every step supported in this system.
"""
STEP_NAMES = {"Sampler": "SA"}

class Step():
    def __init__(self, step_name, df=None, focus_atts=[], sensitive_att=None, target_col=None):

        if df is None:
            print("Input data is empty!")
            raise ValueError

        if sensitive_att:
            if sensitive_att not in df.columns:
                print("Need to specify a valid sensitive attribute!")
                raise ValueError
            self.sensitive_att = sensitive_att
        if target_col is not None:
            if target_col not in df.columns:
                print("Need to specify a valid target attribute to be predicted!")
                raise ValueError
            if len(df[target_col].unique()) != 2:
                print("Only support binary target feature now!")
                raise ValueError
            self.target_col = target_col
            self.pred_target_col = "pred_" + target_col  # store the predicted score (probability) column using this fixed name
        if len(focus_atts) > 0:
            if sum([x not in df.columns for x in focus_atts]) > 0:
                print("Some specified attributes do not appear in the data!")
                raise ValueError
            self.focus_atts = focus_atts
        self.name = step_name
        # self.input_data = df



    def apply(self, df):
        """
        :param df: pandas dataframe, stores the data to apply the learned discretizer.
        :return: pandas dataframe, stores the data after discretize.
        """
        raise NotImplementedError


    def get_name(self): # return full name to print out
        return self.name

    def get_abbr_name(self): # return abbreviated name used in the file name of data
        return STEP_NAMES[self.name]
