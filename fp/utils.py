import pandas as pd

def filter_optimal_results_skyline_order(_df, _order_list):
    _df['selection_rate'] = abs(1 - _df['selection_rate'])
    higher_is_better = ['num_true_positives', 'num_true_negatives', 'num_generalized_true_positives',
                        'num_generalized_true_negatives', 'true_positive_rate', 'true_negative_rate',
                        'generalized_true_positive_rate', 'generalized_true_negative_rate', 'positive_predictive_value',
                        'accuracy', 'num_pred_positives']
    lower_is_better = ['selection_rate', 'num_false_positives', 'num_false_negatives',
                       'num_generalized_false_positives', 'num_generalized_false_negatives', 'false_positive_rate',
                       'false_negative_rate', 'generalized_false_positive_rate', 'generalized_false_negative_rate',
                       'false_discovery_rate', 'false_omission_rate', 'negative_predictive_value', 'error_rate',
                       'num_pred_negatives']
    order = []
    for item in _order_list:
        if item in higher_is_better:
            order.append(False)
        else:
            order.append(True)
    _df = _df.sort_values(_order_list, ascending=order)

    return _df.values[0]


def filter_optimal_results_skyline_formula(_df, _formula):
    df = pd.DataFrame()
    for key in _formula:
        df["norm_" + key] = (_df[key] - _df[key].min()) / (_df[key].max() - _df[key].min())

    df_temp = list(_formula.values())
    keys = list(_formula.keys())
    for col in range(len(keys)):
        keys[col] = "norm_" + keys[col]

    # Multiplying with the multiplier to perform sorting operation
    df['norm_avg'] = df[keys].multiply(df_temp).sum(axis=1)
    frames = [_df, df]
    df_fin = pd.concat(frames, axis=1)

    df_fin = df_fin.sort_values(by='norm_avg', ascending=False)
    cols = [c for c in df_fin.columns if c[:4] != 'norm']
    df_fin = df_fin[cols]
    return df_fin.values[0]