import numpy as np
import pandas as pd
import os, pathlib, itertools
import warnings

warnings.simplefilter('ignore')
os.chdir('..')

from fp.traindata_samplers import CompleteData
from fp.missingvalue_handlers import CompleteCaseAnalysis
from fp.dataset_experiments import GermanCreditDatasetSexExperiment
from fp.scalers import NamedStandardScaler
from fp.learners import LogisticRegression, DecisionTree
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing
from fp.pre_processors import NoPreProcessing, DIRemover

import matplotlib.pyplot as plt
import seaborn as sns

# creating list of parameters that we will alter to observe variations
seeds = [0xbeef, 0xcafe, 0xdead, 0xdeadcafe]
learners = [LogisticRegression()]

processors = [(NoPreProcessing(), NoPostProcessing()), (DIRemover(1.0), NoPostProcessing()), (NoPreProcessing(), RejectOptionPostProcessing())]

# specify the strategy to filter the optimal results on validation set from alll the settings of processors above.
# E.g. if a list ['accuracy', 'selection_rate', 'false_discovery_rate'] is specified, the optimal result is the setting with highest accuracy, selection rate and false discovery rate. If two settings have the same accuracy, the one with highest selection rate is the optimal one, etc. The input list specifies a skyline order to select the optimal one.
# E.g. if a dict {'accuracy': 0.5, 'selection_rate': 0.3, 'false_discovery_rate': 0.2} is specified, the optimal result is the setting with highest values from formula accuracy*0.5+selection_rate*0.3+false_discovery_rate*0.2.
# If more than one settings have the highest value by the above strategies, then all of the settings are returned as optimal.
filter_res_on_val_by_order = ['accuracy', 'selection_rate', 'false_discovery_rate']
filter_res_on_val_by_weight_sum = {'accuracy': 0.5, 'selection_rate': 0.3, 'false_discovery_rate': 0.2}

def calculate_metrics(seed, learners, pre_processors, post_processors, filter_val_strategy):
    '''
        Experiment function to run the experiments with multiple combinations of learners and processors in the input
    '''
    exp = GermanCreditDatasetSexExperiment(
        fixed_random_seed=seed,
        train_data_sampler=CompleteData(),
        missing_value_handler=CompleteCaseAnalysis(),
        numeric_attribute_scaler=NamedStandardScaler(),
        learners=learners,
        pre_processors=pre_processors,
        post_processors=post_processors,
        optimal_validation_strategy=filter_val_strategy)
    exp.run()
    return exp.generate_file_path()

def run_exp(seeds, learners, processors, filter_val_strategy):
    '''
        This is the main driver function that calls the calculate_metrics to give metrices on combinations of various learners, pre and post processing techniques.
    '''
    skyline_res_folder = {}
    for seed in seeds:
        input_preprocessors = [x[0] for x in processors]
        input_postprocessors = [x[1] for x in processors]
        skyline_res_folder[seed] = calculate_metrics(seed, learners, input_preprocessors, input_postprocessors, filter_val_strategy)
    return skyline_res_folder


def output_scatter_plot(f_name, df, x_col, y_col, hue_col='setting', color_p='Set2'):
    '''
        Visualization of the skyline options w.r.t. two metrics (X and Y axis) in the skyline inputs.
    '''
    sns.set(style='whitegrid', font_scale=1.5)
    # add jitters for x to account for ties in the values
    data = df.copy()
    noise_param = 100
    data[x_col] += np.random.random(data.shape[0]) / noise_param - 1 / noise_param / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(13, 6))
    sns.scatterplot(x_col, y_col, hue_col, data=data.query("data == 'val'"), ax=ax1, style='optimal', s=100)
    sns.scatterplot(x_col, y_col, hue_col, data=data.query("data == 'test'"), ax=ax2, style='optimal', s=100)
    ax1.set_title('validation')
    ax2.set_title('test')
    plt.tight_layout()

    # save plot into the disc
    cur_f_path = f_name[0:f_name.rfind("/") + 1]
    if not os.path.exists(cur_f_path):
        directory = os.path.dirname(cur_f_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f_name + '.png')
    plt.close()

def get_skyline_candidates(seed_path_map, focus_seed):
    '''
        Prepare the skyline candidates data for visualization
    '''
    setting_labels = {'reject_option': 'RO', 'diremover-1.0': 'DI1.0',
                      'no_pre_processing': 'NoPre', 'no_post_processing': 'NoPost',
                      'DecisionTree': 'DT', 'LogisticRegression': 'LR'}
    skyline_df = pd.read_csv(seed_path_map[focus_seed] + "skyline_options.csv")
    # only keep the name of preprocessor (idx 1), learner (idx 5), and postprocessor (idx 6) in the settings for visualization purpose
    skyline_df['setting'] = skyline_df['setting'].apply(lambda x: "__".join([x.split('__')[i] for i in range(len(x.split('__'))) if i in [1, 5, 6]]))
    # remove the seed name in the setting for visualization purpose
    skyline_df['setting'] = skyline_df['setting'].apply(lambda x: x.replace('-' + str(focus_seed), ''))
    # rename (shorten) the settings' names for visualization purpose
    skyline_df['setting'] = skyline_df['setting'].apply(lambda x: '_'.join([setting_labels[stepi] for stepi in x.split('__')]))


    # show the candidates using only one fairness intervention method
    return skyline_df[skyline_df['setting'].apply(lambda x: 'NoP' in x)]

# running experiments using above parameters
skyline_order_results = run_exp(seeds, learners, processors, filter_res_on_val_by_order)
print (skyline_order_results)
print ("\n\n\n")

skyline_formula_results = run_exp(seeds, learners, processors, filter_res_on_val_by_weight_sum)
print (skyline_formula_results)


# generate plots for skyline candidates w.r.t the possible combinations of the metrics in the skyline inputs for a single trial, and loop for all the seed trials
for focus_seed in seeds:
    print ('---' * 8,'Generate skyline plots for seed ', str(focus_seed), '---' * 8)
    skyline_order_options = get_skyline_candidates(skyline_order_results, focus_seed)
    skyline_formula_options = get_skyline_candidates(skyline_formula_results, focus_seed)
    for x_col, y_col in itertools.combinations(filter_res_on_val_by_order, 2):
        output_scatter_plot("_".join(['examples/skyline_plots/Order', 's' + str(focus_seed), x_col, y_col]), skyline_order_options, x_col, y_col)
        output_scatter_plot("_".join(['examples/skyline_plots/Formula', 's' + str(focus_seed), x_col, y_col]), skyline_formula_options, x_col, y_col)
        print ('Save skyline order plot for ', x_col, ' and ', y_col, ' in ', "_".join(['examples/skyline_plots/Order', 's' + str(focus_seed), x_col, y_col]), '.png\n')
        print ('Save skyline formula plot for ', x_col, ' and ', y_col, ' in ', "_".join(['examples/skyline_plots/Order', 's' + str(focus_seed), x_col, y_col]), '.png\n')
    print ('---' * 16, 'Done for seed ', str(focus_seed), '---' * 16)
    print ('\n\n')