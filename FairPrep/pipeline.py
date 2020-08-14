"""
    Class of the pipeline, which is the engine of the FairPrep.

"""


import os
from time import time
from pathlib import Path
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")
from FairPrep.preprocess.splitters import *
from FairPrep.preprocess.samplers import *
from FairPrep.preprocess.imputers import *
from FairPrep.preprocess.scalers import *
from FairPrep.preprocess.categorizers import *
from FairPrep.preprocess.encoders import *
from FairPrep.preprocess.fair_preprocessors import *
from FairPrep.model.classifiers import *
from FairPrep.model.fair_classifiers import *
from FairPrep.postprocess.fair_postprocessors import *

from FairPrep.utils import de_dummy_code_df
from FairPrep.utils import dedummy_cols

from aif360.metrics import ClassificationMetric
import json

PRINT_SPLIT = "\n===================================================\n"

# for integrity check of user inputs
PREREQUISITE_STEPS = {"Splitter": [], "Sampler": [], "Imputer": ["Splitter"], "Scaler": ["Imputer"],
                    "Categorizer": ["Imputer"], "Encoder": ["Imputer"], "SpecialEncoder": ["Imputer"],
                    "FairPreprocessor": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"],
                    "Model": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"],
                    "FairInprocessor": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"],
                    "FairPostprocessor": ["Splitter", "Imputer", "Scaler", "Categorizer", "Encoder", "SpecialEncoder"]}
# TODO: read the literature to distinguish TODOS into existing categories.
TODOS = ['generalized_entropy_index', 'between_all_groups_generalized_entropy_index', 'between_group_generalized_entropy_index', 'theil_index', 'coefficient_of_variation',
         'between_group_theil_index', 'between_group_coefficient_of_variation', 'between_all_groups_theil_index', 'between_all_groups_coefficient_of_variation']

LOWER_VALUE_IS_BETTER = ['num_false_positives', 'num_false_negatives', 'num_generalized_false_positives', 'num_generalized_false_negatives', 'false_negative_rate',
                         'generalized_false_positive_rate', 'generalized_false_negative_rate', 'false_discovery_rate', 'false_omission_rate', 'error_rate',
                         'true_positive_rate_difference', 'false_positive_rate_difference', 'false_negative_rate_difference', 'false_omission_rate_difference', 'false_discovery_rate_difference',
                         'average_odds_difference', 'average_abs_odds_difference', 'error_rate_difference', 'statistical_parity_difference'
                         ]
ONE_IS_BEST = ['false_positive_rate_ratio', 'false_negative_rate_ratio', 'false_omission_rate_ratio', 'false_discovery_rate_ratio',
               'error_rate_ratio', 'disparate_impact']


class FairPipeline():
    def __init__(self, data_file_name, target_col, target_positive_values, sensitive_atts, protected_groups, value_mapping, debias_focus_att, seed, sep_flag=None, na_mark=None, verbose=True):
        """

        :param data_file_name: str, file name that stores the data.
        :param *fixed_random_seed: int, the fixed random seed that will be used through the pipeline.
        :param target_col: str, the name of the target variable in above data.
        :param target_positive_values: list of str, each str is the value of the target variable in above data that represents the positive outcome.
        :param sensitive_atts: list, stores the user specified sensitive attributes in above data. Optional.
        :param protected_values: dict, stores the user-specified protected values for above sensitive attributes. Optional.
                                Key is the name in sensitive_atts, value is a list of str, representing the values of the attribute.
                                 Order should mapping to the order in the sensitive_atts.
        """

        self.data_file_name = data_file_name
        self.target_col = target_col
        self.target_positive_values = target_positive_values
        self.target_negative_values = None
        self.sensitive_atts = sensitive_atts
        self.protected_groups = protected_groups
        self.nonprotected_groups = None

        self.value_mapping = {key: float(value) for key, value in value_mapping.items()}
        self.decode_mapping = None
        self.debias_focus_att = debias_focus_att
        self.other_sensitive_atts = None

        self.seed = seed
        self.sep_flag = sep_flag
        self.na_mark = na_mark
        self.verbose = verbose

        self.data_name = None
        self.raw_data = None

        self.encoder_sep = None
        self.attributes = None
        self.metadata_desp = None
        self.num_atts = None
        self.cate_atts = None
        self.pred_target = "pred_" + target_col


        cur_location = os.path.dirname(os.path.realpath(__file__))
        self.project_root_path = cur_location[:cur_location.find("FairPrep")+9]
        self.log_path = 'logs/'
        self.exec_timestamp = self.generate_timestamp()
        self.log_dir_name = None

        self.steps = None

        self.validate_input_parameters()
        self.get_meta_information()

    def validate_input_parameters(self):
        if not os.path.exists(os.path.realpath(self.data_file_name)):
            print("The data you specified doesn't exist!")
            raise ValueError
        if ".csv" not in self.data_file_name:
            print("The data you specified is not valid! Only support .csv file.")
            raise ValueError
        data_name = self.data_file_name.replace(".csv", "")
        self.data_name = data_name[data_name.rfind("/")+1:]

        if self.sep_flag:
            df = pd.read_csv(self.data_file_name, sep=self.sep_flag)
        else: # default ',' separated data
            df = pd.read_csv(self.data_file_name)

        if not df.shape[0]:
            print("Uploaded data is empty!")
            raise ValueError

        # integrity check for target_col
        if self.target_col is None or self.target_positive_values is None:
            print("Need to specify target_col and target_positive_value!")
            raise ValueError

        if self.target_col not in df.columns:
            print("Need to specify a valid target attribute to be predicted!")
            raise ValueError
        target_values = df[self.target_col].unique()
        if len(target_values) != 2:
            print("Only support binary target feature now!")
            raise ValueError
        if len(set(self.target_positive_values).intersection(target_values)) == 0:
            print("Need to specify a valid target positive value!")
            raise ValueError

        self.target_negative_values = list(set(target_values).difference(self.target_positive_values))

        # integrity check for sensitive_atts and protected_values
        if len(set(self.sensitive_atts).difference(df.columns)) > 0:
            print("Some input sensitive attributes are not in the input data!")
            raise ValueError

        if len(set(self.sensitive_atts).difference(self.protected_groups.keys())) > 0:
            print("Some sensitive attributes from the protected groups are not in the specified sensitive attributes!")
            raise ValueError

        if len(set(self.value_mapping.keys()).difference(*[list(df[x].unique()) for x in self.sensitive_atts+[self.target_col]])) > 0:
            print("Some values in the input value mapping are not in the input data!")
            raise ValueError
        if self.debias_focus_att not in self.sensitive_atts or self.debias_focus_att not in df.columns:
            print("The input debias attribute is not in the specified sensitive attributes or is not in the input data!")
            raise ValueError
        other_sensitive_atts = list(set(self.sensitive_atts).difference([self.debias_focus_att]))
        if len(other_sensitive_atts) > 0:
            self.other_sensitive_atts = other_sensitive_atts
        self.raw_data = df
        self.attributes = df.columns

    def get_meta_information(self):
        # infer the numerical and categorical attributes first
        if self.raw_data.describe().shape[0] == 8:# DataFrame.describe() usually returns 8 rows.
            num_atts = set(self.raw_data.describe().columns)
        else:# DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
            num_atts = set()

        cate_atts = set(self.attributes).difference(num_atts)

        self.metadata_desp = {"size": self.raw_data.shape[0], "features": self.raw_data.shape[1],
                              "target attribute": self.target_positive_values,
                              "target positive value": self.target_positive_values,
                              "categorical features": list(cate_atts),
                              "numerical features": list(num_atts)}
        # record the domain of the attributes
        feature_domain = {}
        for attri in self.attributes:
            if attri in cate_atts:
                all_values = self.raw_data[attri].unique()
                feature_domain[attri] = (len(all_values), list(all_values))
            else:
                feature_domain[attri] = (min(self.raw_data[attri]), max(self.raw_data[attri]))
        self.metadata_desp.update({"domain": feature_domain})

        # remove the sensitive and target attributes from the inferred list since these attributes are treated separately in the pipeline
        if self.target_col in cate_atts:
            cate_atts.remove(self.target_col)
        if self.target_col in num_atts:
            num_atts.remove(self.target_col)
        for si in self.sensitive_atts:
            cate_atts.remove(si)

        self.num_atts = list(num_atts)
        self.cate_atts = list(cate_atts)

        log_dir_name = os.path.join(self.project_root_path, self.generate_file_path())
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)

        self.log_dir_name = log_dir_name

        with open(log_dir_name+"/"+self.data_name+'.json', 'w') as outfile:
            json.dump(dict(self.metadata_desp), outfile)

        decode_mapping = {}
        for atti in [self.target_col]+self.sensitive_atts:
            cur_mapping = {}
            for valuei in self.metadata_desp["domain"][atti][1]:
                cur_mapping[self.value_mapping[valuei]] = valuei
            decode_mapping[atti] = cur_mapping
        self.decode_mapping = decode_mapping

        nonprotected_groups = {}
        for atti in self.sensitive_atts:
            nonprotected_groups[atti] = set(self.metadata_desp["domain"][atti][1]).difference([self.protected_groups[atti]]).pop()
        self.nonprotected_groups = nonprotected_groups

        self.metadata_desp.update({"protected groups": self.protected_groups, "nonprotected groups": self.nonprotected_groups})

    def iter_steps(self, steps):
        # TODO: optimize with the input data in the generator function.
        """
        Generator function to iterate steps.
        :param steps: list of objects that represent the steps user want to perform on the input data.
                      Supported steps are listed in STEPS.md.
        :return:  the pandas dataframes that are returned by applying a step on the input data.
        """
        for idx, stepi in enumerate(steps):
            if self.validate_input_steps(idx, steps):
                yield idx, stepi
            else:
                print("The current step cannot be executed and certain steps are required to be executed before this step!")
                raise BrokenPipeError

    def validate_input_steps(self, cur_idx, steps):
        # TODO: move to stage.py for integrity check
        if len(PREREQUISITE_STEPS[steps[cur_idx].step_name()]) == 0:
            # no prerequiresite step
            return True
        elif len(set(PREREQUISITE_STEPS[steps[cur_idx].step_name()]).difference([x.step_name() for x in steps[:cur_idx]])) == 0:
            # all prerequisite steps appear
            return True
        else:
            return False


    def fill_zero_to_dummy_data_before_step(self, input_step, df):
        # The steps that only fit on train and apply to other split data require the same dimension of all the dataset
        # Above requirement is fulfilled by maintaining the dimension of the split data the same as the raw data through auto fill the dummy column for the missing categories

        if input_step.step_name() in ["Model", "FairPreprocessor", "FairInprocessor", "FairPostprocessor"]: # these steps take dummy data as input
            if input_step.step_name() == "FairPreprocessor": # check the missing categories for categorical attributes except the sensitive one that is used in fairness intervention
                if self.other_sensitive_atts:
                    dummy_cols = self.cate_atts + self.other_sensitive_atts
                    encoded_df = pd.get_dummies(df, columns=self.other_sensitive_atts, prefix_sep=self.encoder_sep) # dummy encode the other sensitive attributes that users specify in the inputs
                else:
                    dummy_cols = self.cate_atts
                    encoded_df = df

            elif input_step.step_name() == "Model": # check the missing categories for the sensitive attribute that is used in fairness intervention
                dummy_cols = [self.debias_focus_att]
                encoded_df = pd.get_dummies(df, columns=[self.debias_focus_att], prefix_sep=self.encoder_sep)

            elif input_step.step_name() == "FairInprocessor": # No missing categories checking
                dummy_cols = None
                encoded_df = df

            else: # FairPostprocessor, no missing categories checking
                dummy_cols = None
                if self.debias_focus_att not in df.columns: # need to dedummy the sensitive attribute to be used in fairness intervention
                    encoded_df = dedummy_cols(df, [self.debias_focus_att], sep=self.encoder_sep)

            if dummy_cols: # auto fill the missing categories for the dummy cols
                orig_dummy_cols_n = sum([self.metadata_desp["domain"][x][0] for x in dummy_cols])
                input_dummy_cols_n = len([x for x in encoded_df.columns if x.split(self.encoder_sep)[0] in dummy_cols])
                if input_dummy_cols_n != orig_dummy_cols_n:  # the dimension is not the same due to some missing values in the categorical columns
                    print("==> Auto-filling the missing values after step ", input_step.name(), "!!!")
                    for coli in dummy_cols:
                        for valuei in self.metadata_desp["domain"][coli][1]:
                            if coli in self.sensitive_atts: # the values of the sensitive attributes is numerical after special encoder
                                valuei = self.value_mapping[valuei]
                            if coli + self.encoder_sep + valuei not in encoded_df.columns:
                                print("==> Auto-filling the missing col ", coli + self.encoder_sep + valuei, "!!!")
                                encoded_df.loc[:, coli + self.encoder_sep + valuei] = 0
            return encoded_df

        else: # these steps do not work on dummy data or fit separately on the split data
            # TODO: optimize the below code, i.e., move to different location
            if input_step.name() == "OneHotEncoder": # record the encoding symbol from the encoder
                self.encoder_sep = input_step.sep
            return df

    def run_pipeline(self, steps, save_interdata=False, return_decoded_data=True):
        """

        :param df: pandas dataframe, the data on which steps are applied.
        :param steps: list of classes that represent the steps user want to perform on the above data.
                      Supported steps are listed in STEPS.md.
        :return:  two pandas dataframes: before_data and after_data. after_data is the data after applied the input steps.

        """
        if not steps or not isinstance(steps, list):
            print("Require non-empty list of steps as input!")
            raise ValueError
        if steps[0].step_name() != "Splitter":
            print("First step in preprocess stage must be Splitter!")
            raise ValueError
        self.steps = steps

        # first run the splitter
        print(self.print_log_message(steps, 0))
        steps[0].fit(self.raw_data)
        train_df, val_df, test_df = steps[0].apply(self.raw_data)

        for step_idx, stepi in self.iter_steps(steps):
            if step_idx == 0:
                continue

            print(self.print_log_message(steps, step_idx))

            # run the same step on validation data, if multiple methods are specified at one step, the one with optimal performance on validation set is selected
            if isinstance(steps[step_idx], list): # multiple methods specified at one step
                # TODO: add the selection on validation data

                pass


            else: # single method specified at a step

                # autofill encoded columns after the steps that might affect the dimension of the encoded data. Only for categorical attributes that are not sensitive and target column
                train_df = self.fill_zero_to_dummy_data_before_step(stepi, train_df)
                val_df = self.fill_zero_to_dummy_data_before_step(stepi, val_df)
                test_df = self.fill_zero_to_dummy_data_before_step(stepi, test_df)

                stepi.fit(train_df)
                train_df = stepi.apply(train_df)

                if stepi.fit_only_on_train(): # the steps that fit only on train
                    val_df = stepi.apply(val_df)
                    test_df = stepi.apply(test_df)

                else: # the steps that treat train, val, test independently through the same rule
                    stepi.fit(val_df)
                    val_df = stepi.apply(val_df)

                    stepi.fit(test_df)
                    test_df = stepi.apply(test_df)


            if save_interdata: # save intermediate data on the disc
                for file_i, df_i in zip(["train", "val", "test"], [train_df, val_df, test_df]):
                    self.save_inter_data(file_i, df_i, steps[:step_idx+1])

            print(PRINT_SPLIT)


        # compute the metrics on each split data if there exists the column of the prediction of the target column
        # TODO: optimize for the pipelines that stops at FairPreprocessor
        if len(set(self.get_steps_names()).intersection(["Model", "FairInprocessor", "FairPostprocessor"])) > 0:
            for df_i, flag_i in zip([train_df, val_df, test_df], ["train", "val", "test"]):
                self.compute_eval_metrics(df_i, flag_i)


        # decode the numerical values into original values before return
        if return_decoded_data:
            train_df = self.decode_values(train_df)
            val_df = self.decode_values(val_df)
            test_df = self.decode_values(test_df)


        return train_df, val_df, test_df


    def compute_eval_metrics(self, input_df, data_flag, predict_prob=True):
        # NOTE: the computation is only for a single binary sensitive attribute and binary target attribute
        # NOTE: the data input for BinaryLabelDataset is the data with original values for categorical columns. Can have NULL values that would be dropped inside the wrapper.

        # BinaryLabelDataset requires numerical data
        dummy_df = input_df.copy()
        dummy_df[self.target_col] = dummy_df[self.target_col].astype('float')
        dummy_df[self.pred_target] = dummy_df[self.pred_target].apply(lambda x: float(x >= 0.5))

        aif_df = BinaryLabelDataset(df=dummy_df.drop(columns=[self.pred_target]), label_names=[self.target_col], protected_attribute_names=[self.debias_focus_att],
                                    favorable_label=self.value_mapping[self.target_positive_values[0]],
                                    unfavorable_label=self.value_mapping[self.target_negative_values[0]])

        aif_df_with_pred = aif_df.copy() # to fultill the requirements of AIF 360 metric initialization
        aif_df_with_pred.labels = dummy_df[self.pred_target].values.copy()

        metric = ClassificationMetric(aif_df, aif_df_with_pred,
                                      unprivileged_groups=[{self.debias_focus_att: self.value_mapping[self.protected_groups[self.debias_focus_att]]}],
                                      privileged_groups=[{self.debias_focus_att: self.value_mapping[self.nonprotected_groups[self.debias_focus_att]]}])

        results_df = pd.DataFrame(columns=['PrivilegedStatus', 'MetricName', 'MetricValue'])

        privileged_metric_names = ['num_true_positives', 'num_false_positives', 'num_false_negatives',
                                   'num_true_negatives', 'num_generalized_true_positives',
                                   'num_generalized_false_positives', 'num_generalized_false_negatives',
                                   'num_generalized_true_negatives', 'true_positive_rate', 'false_positive_rate',
                                   'false_negative_rate', 'true_negative_rate', 'generalized_true_positive_rate',
                                   'generalized_false_positive_rate', 'generalized_false_negative_rate',
                                   'generalized_true_negative_rate', 'positive_predictive_value',
                                   'false_discovery_rate', 'false_omission_rate', 'negative_predictive_value',
                                   'accuracy', 'error_rate', 'num_pred_positives', 'num_pred_negatives',
                                   'selection_rate']

        for maybe_privileged in [None, True, False]:
            for metric_name in privileged_metric_names:
                metric_function = getattr(metric, metric_name)
                metric_value = metric_function(privileged=maybe_privileged)
                results_df.loc[results_df.shape[0]]= [maybe_privileged, metric_name, metric_value]

        # TODO: add roc_auc
        # hasattr(model, 'predict_proba')
        # if predict_prob:
        #     auc = roc_auc_score(aif_df.labels, model.predict_proba(aif_df.features)[:, 1])
        # else:
        #     auc = None

        # results_df.loc[results_df.shape[0]] = ['', 'roc_auc', auc]

        global_metric_names = ['true_positive_rate_difference', 'false_positive_rate_difference',
                               'false_negative_rate_difference', 'false_omission_rate_difference',
                               'false_discovery_rate_difference', 'false_positive_rate_ratio',
                               'false_negative_rate_ratio', 'false_omission_rate_ratio',
                               'false_discovery_rate_ratio', 'average_odds_difference', 'average_abs_odds_difference',
                               'error_rate_difference', 'error_rate_ratio', 'disparate_impact',
                               'statistical_parity_difference', 'generalized_entropy_index',
                               'between_all_groups_generalized_entropy_index',
                               'between_group_generalized_entropy_index', 'theil_index', 'coefficient_of_variation',
                               'between_group_theil_index', 'between_group_coefficient_of_variation',
                               'between_all_groups_theil_index', 'between_all_groups_coefficient_of_variation']

        for metric_name in global_metric_names:
            metric_function = getattr(metric, metric_name)
            metric_value = metric_function()
            results_df.loc[results_df.shape[0]] = ['', metric_name, metric_value]

        results_df.to_csv(os.path.join(self.log_dir_name, 'Metrics-{}-{}.csv'.format(self.seed, data_flag)), index=False)
        print("==> Evaluation results are stored in ", os.path.join(self.log_dir_name, 'Metrics-{}-{}.csv'.format(self.seed, data_flag)))

    def filter_optimal_validation_results_on_skyline_input(self):
        # TODO: update according to the latest evaluation file
        """  Identifies the experiment(s) with the highest value as optimal result based on the order specified in the inputs.
            Store the values of all the candidates in a CSV file with name "skyline_options.csv" in the current result dir.
            Keeps the test metrics just for the experiment(s) with the optimal result, i.e., deleting all the non-optimal results.
        """
        results_dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../{}'.format(self.generate_file_path()))
        results_dir = os.listdir(Path(results_dir_name))

        if isinstance(self.optimal_validation_strategy, dict):  # input is a formula stored in a dict
            metrics = list(self.optimal_validation_strategy.keys())
        else:  # input is an order stored in a list
            metrics = self.optimal_validation_strategy

        skyline_options = pd.DataFrame(columns=['setting', 'data'] + metrics)

        # Fetching the values of the input metrics on the validation set of all the experiment settings (e.g. all combinations of learners and processors)
        for result_filename in results_dir:
            file_path = os.path.join(results_dir_name, result_filename)
            result_df = pd.read_csv(file_path)
            result_df.fillna(value='', inplace=True)

            for di in ['val', 'test']:
                skyline_row = [result_filename.replace('.csv',''), di]
                for mi in metrics:
                    mi_value = (result_df.loc[(result_df['Split'] == di) &
                                              (result_df['PrivilegedStatus'] == '') &
                                              (result_df['MetricName'] == mi), 'MetricValue'].values[0])
                    # Reorder the values of metrics if their lower value represents more fair outcome
                    if mi in LOWER_VALUE_IS_BETTER:
                        skyline_row.append(-mi_value)
                    # Transform the values of metrics if 1 represent the fair outcome
                    elif mi in ONE_IS_BEST:
                        if mi > 1:
                            skyline_row.append(1-mi_value)
                        else:
                            skyline_row.append(mi_value-1)
                    else:
                        skyline_row.append(mi_value)
                skyline_options.loc[skyline_options.shape[0]] = skyline_row

        # Normalize (min-max) the values of metrics before computing the skyline order
        for mi in metrics:
            if abs(skyline_options[mi].max()) > 1: # for metrics that have absolute values greater than 1
                skyline_options[mi] = (skyline_options[mi] - skyline_options[mi].min()) / (skyline_options[mi].max() - skyline_options[mi].min())

        # Compute the value based on skyline inputs
        if isinstance(self.optimal_validation_strategy, dict):
            skyline_options['skyline'] = skyline_options.apply(lambda x: sum([x[coli] * weighti for coli, weighti in self.optimal_validation_strategy.items()]), axis=1)
        else:
            # Use exaggerated weights to account ties in the skyline order
            skyline_order_weights = [10**i for i in range(len(self.optimal_validation_strategy)-1, -1, -1)]
            skyline_options['skyline'] = skyline_options.apply(lambda x: sum([x[ci] * weighti for ci, weighti in zip(metrics, skyline_order_weights)]), axis=1)



        # Filter out the optimal setting w.r.t combinations of learners and processors on validation set, i.e., optimal setting is the one with the highest value on validation set
        # If ties appear in the skyline ranked files, all the tied settings are returned as optimal
        skyline_ranked_files = skyline_options[skyline_options['data']=='val'].sort_values('skyline', ascending=False)

        # List of non optimal and optimal filenames to account for ties in the skyline order
        non_optimal_filenames = list()
        optimal_filenames = list()
        max_skyline = max(skyline_ranked_files['skyline'])
        for idx, row in skyline_ranked_files.iterrows():
            if row['skyline'] != max_skyline:
                non_optimal_filenames.append(row['setting'])
            else:
                optimal_filenames.append(row['setting'])

        # Add a column to represent the optimal setting for interpretation and visualization only
        skyline_options['optimal'] = skyline_options.apply(lambda x: int(x['setting'] in optimal_filenames), axis=1)
        skyline_options.to_csv(os.path.join(results_dir_name, 'skyline_options.csv'), index=False)

        # Removing the test results from the non optimal experiment results
        for file_name in non_optimal_filenames:
            file_path = os.path.join(results_dir_name, file_name+".csv")
            result_df = pd.read_csv(file_path)
            result_df = result_df[(result_df['Split'] != 'test')]
            os.remove(file_path)
            result_df.to_csv(file_path, index=False, header=False)

        # Renaming the optimal experiment results file (or files if tie)
        for file_name in optimal_filenames:
            file_path = os.path.join(results_dir_name, file_name+".csv")
            optimal_file_name = '{}{}'.format(file_name, '__OPTIMAL.csv')
            optimal_file_path = os.path.join(results_dir_name, optimal_file_name)
            os.rename(file_path, optimal_file_path)


    def decode_values(self, input_df):
        input_step_names = self.get_steps_names()

        input_df = de_dummy_code_df(input_df)

        if len(set(input_step_names).intersection(["SpecialEncoder", "FairPreprocessor", "Model", "FairInprocessor", "FairPostprocessor"])) > 0:
            # for the sensitive and target attributes
            for atti in [self.target_col] + self.sensitive_atts:
                input_df[atti] = input_df[atti].astype('float')
                input_df[atti] = input_df[atti].apply(lambda x: self.decode_mapping[atti][x])

        if len(set(input_step_names).intersection(["Model", "FairInprocessor", "FairPostprocessor"])) > 0:
            # for the prediction column
            if self.pred_target not in input_df.columns:
                print("No prediction columns! Error in pipeline!")
                raise BrokenPipeError
            input_df[self.pred_target] = input_df[self.pred_target].astype('float')
            input_df[self.pred_target] = input_df[self.pred_target].apply(lambda x: float(x >= 0.5))  # for the model that returns a score instead of labels
            input_df[self.pred_target] = input_df[self.pred_target].apply(lambda x: self.decode_mapping[self.target_col][x])

        return input_df

    def get_steps_names(self):
        return [x.step_name() for x in self.steps]

    def print_log_message(self, steps, step_idx):
        if not self.verbose:
            return None
        return '(step %d of %d) running %s' % (step_idx + 1, len(steps), steps[step_idx].name()) + PRINT_SPLIT

    def get_executed_steps_name(self, executed_steps):
        return "_".join([x.abbr_name() for x in executed_steps])

    def generate_file_path(self, file_name=''):
        dir_name = '{}__{}/'.format(self.exec_timestamp, self.data_name)
        return self.log_path + dir_name + file_name

    def generate_timestamp(self):
        return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]


    def save_inter_data(self, file_name, df, executed_steps):
        data_file_name = '{}-{}-{}.csv'.format(self.get_executed_steps_name(executed_steps), self.seed, file_name)
        data_file_path = os.path.join(self.log_dir_name, data_file_name)

        df.to_csv(data_file_path, index=False)
        if self.verbose:
            print ("Data is saved to ", data_file_path)

    

if __name__ == '__main__':

    data_file = "../data/german_AIF_test.csv"
    y_col = "credit"
    y_posi = ["good"]
    sensi_atts = ["age", "sex"]
    protected_groups = {"age":"young", "sex": "female"}
     # NOTE that the protected group (unprivileged group) usually is decoded into 0
    value_mapping = {"female": 0, "male": 1, "good": 1, "bad": 0, "young": 0, "old": 1}

    debias_focus_att = "sex"
    global_seed = 0

    numerical_atts = ["month", "credit_amount"]
    categorical_atts = ["status", "employment", "housing"]


    pipeline = [RandomSplitter([0.5, 0.3, 0.2], global_seed), NoSampler(), NoImputer(), SK_MinMaxScaler(numerical_atts),
                  NoBinarizer(),  OneHotEncoder(categorical_atts), MappingEncoder([y_col] + sensi_atts, value_mapping),
                  AIF_Reweighing(y_col, debias_focus_att), OPT_LogisticRegression(y_col, global_seed), NoFairPostprocessor()]



    cur_pip = FairPipeline(data_file, y_col, y_posi, sensi_atts, protected_groups, value_mapping, debias_focus_att, global_seed)


    train_now, val_now, test_now = cur_pip.run_pipeline(pipeline, save_interdata=True)
    train_now.to_csv(cur_pip.log_dir_name+"Final_train.csv", index=False)
