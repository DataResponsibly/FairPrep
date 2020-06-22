from sklearn.metrics import confusion_matrix
import numpy as np
def get_static_label(df, sensi_atts, target_name, round_digit=3):
    groupby_cols = sensi_atts+[target_name]
    placeholder_att = list(set(df.columns).difference(groupby_cols))[0]
    
    count_all = df[groupby_cols+[placeholder_att]].groupby(groupby_cols).count()
    values_all = count_all.get_values()
    index_all = list(count_all.index)

    if len(sensi_atts) == 1:
        norm_cols = [target_name]
    elif len(sensi_atts) == 2:
        norm_cols = [sensi_atts[0], target_name]
    norm_values = df[norm_cols+[placeholder_att]].groupby(norm_cols).count().get_values()
    
    res_dict = {}
    if 0 < len(sensi_atts) <= 2:
        s1_n = len(df[sensi_atts[0]].unique())
        t_n = len(df[target_name].unique())
        for idx, tuple_i in enumerate(index_all):
            if len(tuple_i[:-1]) == 1:
                key_tuple = (tuple_i[0])
            else:
                key_tuple = tuple_i[:-1]
            idx_denom = idx % 2 + int(idx / (s1_n*t_n))*t_n # only work for binary 2nd sensitive att
            if key_tuple not in res_dict:
                res_dict[key_tuple] = {tuple_i[-1]: round(values_all[idx][0]/norm_values[idx_denom][0], round_digit)}
            else:
                res_dict[key_tuple].update({tuple_i[-1]: round(values_all[idx][0]/norm_values[idx_denom][0], round_digit)})
    else: # for more than 2 sensitive atts
        pass
    return res_dict

def compute_evaluation_metric_binary(true_y, pred_y, label_order):
    TN, FP, FN, TP = confusion_matrix(true_y, list(pred_y), labels=label_order).ravel()
    P = TP + FN
    N = TN + FP
    ACC = (TP+TN) / (P+N) if (P+N) > 0.0 else np.float64(0.0)
    return dict(
                PR = P/ (P+N), P = TP + FN, N = TN + FP,
                TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
                PPV=TP / (TP+FP) if (TP+FP) > 0.0 else np.float64(0.0),
                NPV=TN / (TN+FN) if (TN+FN) > 0.0 else np.float64(0.0),
                FDR=FP / (FP+TP) if (FP+TP) > 0.0 else np.float64(0.0),
                FOR=FN / (FN+TN) if (FN+TN) > 0.0 else np.float64(0.0),
                ACC=ACC,
                ERR=1-ACC,
                F1=2*TP / (2*TP+FP+FN) if (2*TP+FP+FN) > 0.0 else np.float64(0.0)
            )
def get_performance_label(df, sensi_atts, target_name, posi_target, output_metrics=["TPR", "FPR", "TNR", "FNR", "PR"], round_digit=3):
    
    groupby_cols = sensi_atts+[target_name]
    placeholder_att = list(set(df.columns).difference(groupby_cols))[0]
    
    count_all = df[groupby_cols+[placeholder_att]].groupby(groupby_cols).count()
    index_all = list(count_all.index)

    res_dict = {}
    target_label_order = [posi_target, set(df[target_name]).difference([posi_target]).pop()]
    
    for tuple_i in index_all:
        if len(tuple_i[:-1]) == 1:
            key_tuple = (tuple_i[0])
        else:
            key_tuple = tuple_i[:-1]
        cur_q = []
        for idx, vi in enumerate(tuple_i[:-1]):
            cur_q.append("{}=='{}'".format(sensi_atts[idx], vi))
        tuple_df = df.query(" and ".join(cur_q))
        metrics_all = compute_evaluation_metric_binary(list(tuple_df[target_name]), list(tuple_df["pred_"+target_name]), target_label_order)
        res_dict[key_tuple] = {x: round(metrics_all[x], round_digit) for x in metrics_all if x in output_metrics}
        
    return res_dict