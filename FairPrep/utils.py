"""
    Util functions in FairPrep. Some are developed by AIFairness 360 StructuredDataset (https://github.com/Trusted-AI/AIF360)
"""
from collections import defaultdict
import pandas as pd

def de_dummy_code_df_cols(df, col_names, sep="=", set_category=True):
    """De-dummy some columns a dummy-coded dataframe obtained with pd.get_dummies().
    """
    feature_names_dum_d, feature_names_nodum = parse_feature_names(col_names)
    df_new = pd.DataFrame(index=df.index,
        columns=feature_names_nodum + list(feature_names_dum_d.keys()))

    for fname in feature_names_nodum:
        df_new[fname] = df[fname].values.copy()

    for fname, vl in feature_names_dum_d.items():
        for v in vl:
            df_new.loc[df[fname+sep+str(v)] == 1, fname] = str(v)

    if set_category:
        for fname in feature_names_dum_d.keys():
            df_new[fname] = df_new[fname].astype('category')

    return df_new

def dedummy_cols(df, dedummy_atts, sep="="):
    encoded_att_cols = []
    for coli in df.columns:
        if coli.split(sep)[0] in dedummy_atts:
            encoded_att_cols.append(coli)
    return pd.concat([df.drop(columns=encoded_att_cols), de_dummy_code_df_cols(df, encoded_att_cols, sep=sep)], axis=1)


"""
    Util functions from AIFairness 360 StructuredDataset (https://github.com/Trusted-AI/AIF360)
"""
def parse_feature_names(feature_names, sep="="):
    """Parse feature names to ordinary and dummy coded candidates.

    Args:
        feature_names (list): Names of features
        sep (char): Separator to designate the dummy coded category in the
            feature name

    Returns:
        (dict, list):

            * feature_names_dum_d (dict): Keys are the base feature names
              and values are the categories.

            * feature_names_nodum (list): Non-dummy coded feature names.

    Examples:
        >>> feature_names = ["Age", "Gender=Male", "Gender=Female"]
        >>> StructuredDataset._parse_feature_names(feature_names, sep="=")
        (defaultdict(<type 'list'>, {'Gender': ['Male', 'Female']}), ['Age'])
    """
    feature_names_dum_d = defaultdict(list)
    feature_names_nodum = list()
    for fname in feature_names:
        if sep in fname:
            fname_dum, v = fname.split(sep, 1)
            feature_names_dum_d[fname_dum].append(v)
        else:
            feature_names_nodum.append(fname)

    return feature_names_dum_d, feature_names_nodum
def de_dummy_code_df(df, sep="=", set_category=False):
    """De-dummy code a dummy-coded dataframe obtained with pd.get_dummies().

    After reversing dummy coding the corresponding fields will be converted
    to categorical.

    Args:
        df (pandas.DataFrame): Input dummy coded dataframe
        sep (char): Separator between base name and dummy code
        set_category (bool): Set the de-dummy coded features
                to categorical type

    Examples:
        >>> columns = ["Age", "Gender=Male", "Gender=Female"]
        >>> df = pd.DataFrame([[10, 1, 0], [20, 0, 1]], columns=columns)
        >>> _de_dummy_code_df(df, sep="=")
           Age  Gender
        0   10    Male
        1   20  Female
    """

    feature_names_dum_d, feature_names_nodum = parse_feature_names(df.columns)
    df_new = pd.DataFrame(index=df.index,
        columns=feature_names_nodum + list(feature_names_dum_d.keys()))

    for fname in feature_names_nodum:
        df_new[fname] = df[fname].values.copy()

    for fname, vl in feature_names_dum_d.items():
        for v in vl:
            df_new.loc[df[fname+sep+str(v)] == 1, fname] = str(v)

    if set_category:
        for fname in feature_names_dum_d.keys():
            df_new[fname] = df_new[fname].astype('category')

    return df_new

