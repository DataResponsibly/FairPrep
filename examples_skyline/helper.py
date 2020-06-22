import os 
import sys
import shutil
import pandas as pd
import numpy as np

path = 'logs'
def extract_info(learner_acc, learner_di, learner_fnr, learner_fpr, learner_label, learner_optimal):
    try:
        for folder_name, sub_folders, file_names in os.walk(path):
            for sub_folder in sub_folders:
                file_list = os.listdir(os.path.join(path, sub_folder))
                file_list.sort()
                for file in file_list:
                    file_path = os.path.join(path,sub_folder, file)
                    df = pd.read_csv(str(file_path), header=None, names = ['split_type', 'label', 'parameter', 'value'])
                    test_data = df['split_type'] == 'test'
                    
                    flags = file.split("__")
                    pre_processor = flags[1]
                    post_processor = flags[2]
 
                    if(len(flags) == 8 and flags[-1] == "OPTIMAL.csv"):
                        learner_optimal.append(True)
                    else:
                        learner_optimal.append(False)
                        
                    if(post_processor == "no_post_processing"):
                        learner_label.append(pre_processor)
                    else:
                        learner_label.append(post_processor[:17])

                    di = (df['parameter'] == 'disparate_impact') & test_data
                    di_value = df.loc[di, :]

                    learner_di.append(di_value.iloc[0, -1])
                    
                    acc = (df['parameter'] == 'accuracy') & test_data
                    acc_value = df.loc[acc, :]
                    learner_acc.append(acc_value.iloc[0, -1])

                    fnr = (df['parameter'] == 'generalized_false_negative_rate') & test_data
                    fnr_value = df.loc[fnr, :]
                    learner_fnr.append(fnr_value.iloc[0, -1])

                    fpr = (df['parameter'] == 'generalized_false_positive_rate') & test_data
                    fpr_value = df.loc[fpr, :]
                    learner_fpr.append(fpr_value.iloc[0, -1])

                shutil.rmtree(path + '/' + sub_folder)
    except:
        pass

