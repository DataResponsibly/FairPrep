import os
import numpy as np
import pandas as pd

from time import time
from pathlib import Path
from datetime import datetime
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class BinaryClassificationExperiment:


    def __init__(self,
                 fixed_random_seed,
                 test_set_ratio,
                 validation_set_ratio,
                 label_name,
                 positive_label,
                 numeric_attribute_names,
                 categorical_attribute_names,
                 attributes_to_drop_names,
                 train_data_sampler,
                 missing_value_handler,
                 numeric_attribute_scaler,
                 learners,
                 pre_processors,
                 post_processors,
                 protected_attribute_names,
                 privileged_classes,
                 privileged_groups,
                 unprivileged_groups,
                 dataset_metadata,
                 dataset_name):

        self.fixed_random_seed = fixed_random_seed
        self.test_set_ratio = test_set_ratio
        self.validation_set_ratio = validation_set_ratio
        self.label_name = label_name
        self.positive_label = positive_label
        self.numeric_attribute_names = numeric_attribute_names
        self.categorical_attribute_names = categorical_attribute_names
        self.attributes_to_drop_names = attributes_to_drop_names
        self.train_data_sampler = train_data_sampler
        self.missing_value_handler = missing_value_handler
        self.numeric_attribute_scaler = numeric_attribute_scaler
        self.learners = learners
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.protected_attribute_names = protected_attribute_names
        self.privileged_classes = privileged_classes
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.dataset_metadata = dataset_metadata
        self.dataset_name = dataset_name
        self.log_path = 'logs/'
        self.exec_timestamp = self.generate_timestamp()



    # --- Helper Methods Begin ------------------------------------------------


    def unique_file_name(self, learner, pre_processor, post_processor):
        return '{}__{}__{}__{}__{}__{}__{}'.format(self.dataset_name,
                                                   learner.name(),
                                                   self.missing_value_handler.name(),
                                                   self.train_data_sampler.name(),
                                                   self.numeric_attribute_scaler.name(),
                                                   pre_processor.name(),
                                                   post_processor.name())


    def generate_file_path(self, file_name=''):
        dir_name = '{}_{}/'.format(self.exec_timestamp, self.dataset_name)
        return self.log_path + dir_name + file_name


    def generate_timestamp(self):
        return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]


    def load_raw_data(self):
        raise NotImplementedError


    def learn_classifier(self, learner, annotated_train_data, fixed_random_seed):
        return learner.fit_model(annotated_train_data, fixed_random_seed)


    def preprocess_data(self, pre_processor, annotated_dataset):
        return pre_processor.pre_process(annotated_dataset, self.privileged_groups, self.unprivileged_groups)


    def post_process_predictions(self, post_processor, validation_dataset, validation_dataset_with_predictions,
                                 testset_with_predictions):
        return post_processor.post_process(validation_dataset, validation_dataset_with_predictions,
                                                testset_with_predictions, self.fixed_random_seed,
                                                self.privileged_groups, self.unprivileged_groups)


    def apply_model(self, data, scalers, adjusted_annotated_train_data, pre_processor, learner, model):
        filtered_data = self.missing_value_handler.handle_missing(data)
        print(self.missing_value_handler.name(), 'removed', len(data) - len(filtered_data),
              'instances from validation data')

        for numerical_attribute, scaler in scalers.items():
            numerical_attribute_data = np.array(filtered_data[numerical_attribute]).reshape(-1, 1)
            scaled_numerical_attribute_data = scaler.transform(numerical_attribute_data)
            filtered_data.loc[:, numerical_attribute] = scaled_numerical_attribute_data

        annotated_data = StandardDataset(
            df=filtered_data,
            label_name=self.label_name,
            favorable_classes=[self.positive_label],
            protected_attribute_names=self.protected_attribute_names,
            privileged_classes=self.privileged_classes,
            categorical_features=self.categorical_attribute_names,
            features_to_drop=self.attributes_to_drop_names,
            metadata=self.dataset_metadata
        )

        adjusted_annotated_data = self.preprocess_data(pre_processor, annotated_data)

        train_feature_names = adjusted_annotated_train_data.feature_names
        current_feature_names = adjusted_annotated_data.feature_names

        feature_names_in_train_but_not_in_current = set(train_feature_names).difference(
            set(current_feature_names))

        print("Injecting zero columns for features not present", feature_names_in_train_but_not_in_current)

        validation_data_df, _ = adjusted_annotated_data.convert_to_dataframe()

        for feature_name in feature_names_in_train_but_not_in_current:
            validation_data_df.loc[:, feature_name] = 0.0

        adjusted_annotated_data.feature_names = train_feature_names
        adjusted_annotated_data.features = validation_data_df[train_feature_names].values.copy()

        adjusted_annotated__data_with_predictions = adjusted_annotated_data.copy()

        if learner.needs_annotated_data_for_prediction():
            adjusted_annotated__data_with_predictions = model.predict(adjusted_annotated_data)
        else:
            adjusted_annotated__data_with_predictions.labels = model.predict(adjusted_annotated_data.features)

            try:
                class_probs = model.predict_proba(adjusted_annotated_data.features)
                adjusted_annotated__data_with_predictions.scores = class_probs[:, 0]
            except AttributeError:
                print("WARNING: MODEL CANNOT ASSIGN CLASS PROBABILITIES")

        return adjusted_annotated_data, adjusted_annotated__data_with_predictions


    def log_metrics(self, results_file, model, annotated_data, annotated_data_with_predictions, prefix):
        metric = ClassificationMetric(annotated_data, annotated_data_with_predictions,
                                      unprivileged_groups=self.unprivileged_groups,
                                      privileged_groups=self.privileged_groups)

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
                results_file.write('{},{},{},{}\n'.format(prefix, maybe_privileged, metric_name, metric_value))

        if hasattr(model, 'predict_proba'):
            auc = roc_auc_score(annotated_data.labels, model.predict_proba(annotated_data.features)[:, 1])
        else:
            auc = None

        results_file.write('{},,roc_auc,{}\n'.format(prefix, auc))

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
            results_file.write('{},,{},{}\n'.format(prefix, metric_name, metric_value))


    # --- Helper Methods End --------------------------------------------------

    
    def run_single_exp(self, annotated_train_data, validation_data, test_data, scalers, pre_processor,
                       learner, post_processor):
        """Executes a single instance of experiment out of all the possible 
        experiments from the given parameters.
        
        Parameters:
        -----------
        annotated_train_data : annotated aif360.datasets.StandardDataset of 
            train data

        validation_data : pandas dataframe of validation data

        test_data : pandas dataframe of test data

        scalers : dictionary with (key='feature name', value='type of scaler')

        pre_processor : fairprep pre-processor abstraction from 
            aif360.algorithms.pre_processors
    
        learner : fairprep learner abstraction from sci-kit learn or 
            aif360.algorithms.inprocessing
    
        post_processor : fairprep pre-processor abstraction from 
            aif360.algorithms.post_processors
        """
        
        adjusted_annotated_train_data = self.preprocess_data(pre_processor, annotated_train_data)
        model = self.learn_classifier(learner, adjusted_annotated_train_data, self.fixed_random_seed)

        adjusted_annotated_train_data_with_predictions = adjusted_annotated_train_data.copy()

        if learner.needs_annotated_data_for_prediction():
            adjusted_annotated_train_data_with_predictions = model.predict(
                adjusted_annotated_train_data_with_predictions)
        else:
            adjusted_annotated_train_data_with_predictions.labels = model.predict(
                adjusted_annotated_train_data_with_predictions.features)

        adjusted_annotated_validation_data, adjusted_annotated_validation_data_with_predictions = \
            self.apply_model(validation_data, scalers, adjusted_annotated_train_data, pre_processor, learner, model)

        adjusted_annotated_test_data, adjusted_annotated_test_data_with_predictions = \
            self.apply_model(test_data, scalers, adjusted_annotated_train_data, pre_processor, learner, model)

        adjusted_annotated_test_data_with_predictions = self.post_process_predictions(post_processor,
            adjusted_annotated_validation_data,
            adjusted_annotated_validation_data_with_predictions,
            adjusted_annotated_test_data_with_predictions)

        results_file_name = '../{}{}-{}.csv'.format(
            self.generate_file_path(), self.unique_file_name(pre_processor, learner, post_processor), self.fixed_random_seed)
        results_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_file_name)
        
        results_dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../{}'.format(self.generate_file_path()))
        if not os.path.exists(results_dir_name):
            os.makedirs(results_dir_name)

        with open(results_file_path, 'w') as results_file:

            self.log_metrics(results_file, model, adjusted_annotated_validation_data,
                             adjusted_annotated_validation_data_with_predictions, 'val')
            self.log_metrics(results_file, model, adjusted_annotated_train_data,
                             adjusted_annotated_train_data_with_predictions, 'train')
            self.log_metrics(results_file, model, adjusted_annotated_test_data,
                             adjusted_annotated_test_data_with_predictions, 'test')


    def filter_optimal_results(self):
        """Identifies the experiment(s) with the highest accuracy as optimal 
            result. Keeps the test metrics just for the experiment(s) with the
            optimal result.
        """
        
        results_dir = os.listdir(Path(self.generate_file_path()))
        accuracies = dict()
        max_accuracy = 0

        # Fetching the accuracy from the row('val', 'None', 'accuracy') of all the experiment results
        for result_filename in results_dir:
            file_path = self.generate_file_path(result_filename)
            result_df = pd.read_csv(file_path, header=None, names=['split', 'maybe_privileged', 'metric_name', 'metric_value'])
            accuracy = (result_df.loc[(result_df['split'] == 'val') & 
                                      (result_df['maybe_privileged'] == 'None') & 
                                      (result_df['metric_name'] == 'accuracy'), 'metric_value'].values[0])
            accuracies[result_filename] = accuracy
            if accuracy > max_accuracy:
                max_accuracy = accuracy

        # List of non optimal and optimal filenames and accuracy
        non_optimal_filenames = list()
        optimal_filenames = list()
        for filename, accuracy in accuracies.items():
            if accuracy != max_accuracy:
                non_optimal_filenames.append(filename)
            else:
                optimal_filenames.append(filename)

        # Removing the test results from the non optimal experiment results      
        for file_name in non_optimal_filenames:
            file_path = self.generate_file_path(file_name)
            result_df = pd.read_csv(file_path, header=None, names=['split', 'maybe_privileged', 'metric_name', 'metric_value'])
            result_df = result_df[(result_df['split'] != 'test')]
            os.remove(file_path)
            result_df.to_csv(file_path, index=False, header=False)

        # Renaming the optimal experiment results file (or files if tie) 
        for file_name in optimal_filenames:
            file_path = self.generate_file_path(file_name)
            optimal_file_name = file_name[:-4] + '__OPTIMAL.csv'
            optimal_file_path = self.generate_file_path(optimal_file_name)
            os.rename(file_path, optimal_file_path)


    def run(self):
        """Executes all the possible experiments from the combination of  given
            learners, pre-processors and post-processors.
            
            No. of experiments = (#learners * #preprocessors * #postprocessors)
        """
        np.random.seed(self.fixed_random_seed)

        data = self.load_raw_data()

        all_train_data, test_and_validation_data = train_test_split(data, test_size=self.test_set_ratio +
                                                                    self.validation_set_ratio,
                                                                    random_state=self.fixed_random_seed)

        train_data = self.train_data_sampler.sample(all_train_data)

        second_split_ratio = self.test_set_ratio / (self.test_set_ratio + self.validation_set_ratio)

        validation_data, test_data = train_test_split(test_and_validation_data, test_size=second_split_ratio,
                                                      random_state=self.fixed_random_seed)

        self.missing_value_handler.fit(train_data)
        filtered_train_data = self.missing_value_handler.handle_missing(train_data)

        print(self.missing_value_handler.name(), 'removed', len(train_data) - len(filtered_train_data),
              'instances from training data')

        scalers = {}

        for numerical_attribute in self.numeric_attribute_names:
            numerical_attribute_data = np.array(filtered_train_data[numerical_attribute]).reshape(-1, 1)
            scaler = clone(self.numeric_attribute_scaler).fit(numerical_attribute_data)
            scaled_numerical_attribute_data = scaler.transform(numerical_attribute_data)

            filtered_train_data.loc[:, numerical_attribute] = scaled_numerical_attribute_data
            scalers[numerical_attribute] = scaler

        annotated_train_data = StandardDataset(
            df=filtered_train_data,
            label_name=self.label_name,
            favorable_classes=[self.positive_label],
            protected_attribute_names=self.protected_attribute_names,
            privileged_classes=self.privileged_classes,
            categorical_features=self.categorical_attribute_names,
            features_to_drop=self.attributes_to_drop_names,
            metadata=self.dataset_metadata
        )

        for pre_processor in self.pre_processors:
            for learner in self.learners:
                for post_processor in self.post_processors:
                    self.run_single_exp(annotated_train_data, validation_data, test_data, scalers, 
                                        pre_processor, learner, post_processor)
        self.filter_optimal_results()
