import os
import sys
import shutil
import warnings
warnings.simplefilter('ignore')

from helper import extract_info

sys.path.append(os.getcwd())

from fp.traindata_samplers import CompleteData
from fp.missingvalue_handlers import CompleteCaseAnalysis, ModeImputer, DataWigSimpleImputer
from fp.dataset_experiments import AdultDatasetWhiteExperiment
from fp.scalers import NamedStandardScaler, NamedMinMaxScaler
from fp.learners import NonTunedLogisticRegression, LogisticRegression, DecisionTree, NonTunedLogisticRegression, NonTunedDecisionTree          
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing
from fp.pre_processors import NoPreProcessing, DIRemover, Reweighing

import numpy as np
import itertools
import matplotlib.pyplot as plt

#creating list of parameters that we will alter to observe variations
seeds = [0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe, 0xcafebeef, 50, 583, 5278, 100000, 0xefac,0xfeeb, 0xdaed, 0xefacdaed, 0xfeebdead]
learners = [NonTunedLogisticRegression(), LogisticRegression(), NonTunedDecisionTree(), DecisionTree()]
processors = [(NoPreProcessing(), NoPostProcessing()), (DIRemover(1.0), NoPostProcessing()), (DIRemover(0.5), NoPostProcessing()), (Reweighing(), NoPostProcessing()),
              (NoPreProcessing(), RejectOptionPostProcessing()), (NoPreProcessing(), CalibratedEqualOddsPostProcessing())]
impute_column_list = ['workclass', 'occupation', 'native-country']
label_column = 'income-per-year'
datawig_imputer = DataWigSimpleImputer(impute_column_list, label_column,out='out')
missing_value_imputers = [CompleteCaseAnalysis(),  ModeImputer(impute_column_list), datawig_imputer]


def calculate_metrics(seed, learner, missing_value_imputer,pre_processor,post_processor):
    '''
    Experiment function to run the experiments
    '''
    exp = AdultDatasetWhiteExperiment( 
        fixed_random_seed=seed,
        train_data_sampler=CompleteData(),
        missing_value_handler=missing_value_imputer,
        numeric_attribute_scaler=NamedStandardScaler(),
        learners=[learner],
        pre_processors=[pre_processor],
        post_processors=[post_processor])
    exp.run()

def run_exp(seeds, learners, processors):
    '''
    This is the main driver function that calls the calculate_metrics to give metrices on combinations of various learners, pre and post processing techniques.
    '''
    accuracy, disp_imp, fnr, fpr = [], [], [], []
    for learner in learners:
        for processor in processors:
            for imputer in missing_value_imputers:
                imputer_acc, imputer_di, imputer_fnr, imputer_fpr = [], [], [], []
                for seed in seeds:    
                    calculate_metrics(seed, learner, imputer, pre_processor=processor[0], post_processor=processor[1])
                    extract_info(imputer_acc, imputer_di, imputer_fnr, imputer_fpr)
                accuracy.append(imputer_acc)
                disp_imp.append(imputer_di)
                fnr.append(imputer_fnr)
                fpr.append(imputer_fpr)
    return accuracy, disp_imp, fnr, fpr

accuracy, disp_imp, fnr, fpr  = run_exp(seeds, learners, processors)

def plotter(x, y, x_ticks, x_label, main_title):
    '''
    Function to plot various comparison plots.
    '''
    learner_list = ['NonTunedLogistic','TunedLogistic', 'NonTunedDecisionTree', 'TunedDecisionTree']
    processor_list = ['NoPreProcessing', 'DIRemover(1.0)', 'DIRemover(0.5)', 'Reweighing', 'Reject Option', 'Caliberated Equal Odds']
    title_list = list(itertools.product(learner_list,processor_list))
    label_list = [('CompleteCase',  'ModeImputer', 'datawig_simple')]
    fig, axs = plt.subplots(len(learner_list), len(processor_list), figsize=((10,20)))
    axs = axs.flatten()
    for i in range(0, len(y), 3):
        loc = i//3
        axs[loc].scatter(x[i], y[i], c='b', marker='o')
        axs[loc].scatter(x[i+1], y[i+1], c='r', marker='o')
        axs[loc].scatter(x[i+2], y[i+2], c='g', marker='o')
        axs[loc].set_xticks(x_ticks)
        axs[loc].set_yticks(np.arange(0.5, 1, 0.1))
        axs[loc].set_title(title_list[i//3],fontsize=8)

        axs[loc].grid(True)
        axs[loc].set_xlabel(x_label)
        axs[loc].set_ylabel('Accuracy')
        axs[loc].legend(label_list[int(i%3/2)])
    fig.suptitle(main_title)
    plt.subplots_adjust(wspace=0.3, hspace=0.43)
    fig.savefig('examples/' + main_title + '.png')
    plt.show()

plotter(x=disp_imp, y=accuracy, x_ticks=[0.5, 1, 1.5], x_label='DI', main_title='missing_data_accuracy_vs_di')
plotter(x=fnr, y=accuracy, x_ticks=[-0.4, 0, 0.4], x_label='FNR', main_title='missing_data_accuracy_vs_fnr')
plotter(x=fpr, y=accuracy, x_ticks=[-0.4, 0, 0.4], x_label='FPR', main_title='missing_data_accuracy_vs_fpr')

