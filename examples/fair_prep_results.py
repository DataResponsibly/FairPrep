import os
import sys
import shutil
import warnings
warnings.simplefilter('ignore')

from helper import extract_info

sys.path.append(os.getcwd())

from fp.traindata_samplers import CompleteData
from fp.missingvalue_handlers import CompleteCaseAnalysis
from fp.dataset_experiments import GermanCreditDatasetSexExperiment
from fp.scalers import NamedStandardScaler, NamedMinMaxScaler
from fp.learners import NonTunedLogisticRegression, LogisticRegression, DecisionTree, NonTunedLogisticRegression, NonTunedDecisionTree          
from fp.post_processors import NoPostProcessing, RejectOptionPostProcessing, EqualOddsPostProcessing, CalibratedEqualOddsPostProcessing
from fp.pre_processors import NoPreProcessing, DIRemover, Reweighing

import numpy as np
import matplotlib.pyplot as plt

#creating list of parameters that we will alter to observe variations
seeds = [0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe, 0xcafebeef, 50, 583, 5278, 100000, 0xefac,0xfeeb, 0xdaed, 0xefacdaed, 0xfeebdead]
learners = [NonTunedLogisticRegression(), LogisticRegression(), NonTunedDecisionTree(), DecisionTree()]
processors = [(NoPreProcessing(), NoPostProcessing()), (DIRemover(1.0), NoPostProcessing()), (DIRemover(0.5), NoPostProcessing()), (Reweighing(), NoPostProcessing()),
              (NoPreProcessing(), RejectOptionPostProcessing()), (NoPreProcessing(), CalibratedEqualOddsPostProcessing())]

def calculate_metrics(seed, learner, pre_processor, post_processor):
    '''
    Experiment function to run the experiments
    '''
    exp = GermanCreditDatasetSexExperiment(
        fixed_random_seed=seed,
        train_data_sampler=CompleteData(),
        missing_value_handler=CompleteCaseAnalysis(),
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
    for processor in processors:
        for learner in learners:
            learner_acc, learner_di, learner_fnr, learner_fpr = [], [], [], []
            for seed in seeds:    
                calculate_metrics(seed, learner, pre_processor=processor[0], post_processor=processor[1])
                learner_acc, learner_di, learner_fnr, learner_fpr = extract_info(learner_acc, learner_di, learner_fnr, learner_fpr)
                print(learner_acc)
            accuracy.append(learner_acc)
            disp_imp.append(learner_di)
            fnr.append(learner_fnr)
            fpr.append(learner_fpr)
    
    return accuracy, disp_imp, fnr, fpr

accuracy, disp_imp, fnr, fpr  = run_exp(seeds, learners, processors)

def plotter(x, y, x_ticks, x_label, main_title):
    '''
    Function to plot various comparison plots.
    '''
    title_list = ['NoPreProcessing', 'DIRemover(1.0)', 'DIRemover(0.5)', 'Reweighing', 'Reject Option', 'Caliberated Equal Odds']
    label_list = [('NonTunedLogistic', 'TunedLogistic'), ('NonTunedDecisionTree', 'TunedDecisionTree')] 
    fig, axs = plt.subplots(6, 2, figsize=((10,20)))
    axs = axs.flatten()
    for i in range(0, len(y), 2):
        loc = i//2
        axs[loc].scatter(x[i], y[i], c='b', marker='o')
        axs[loc].scatter(x[i+1], y[i+1], c='r', marker='o')
        axs[loc].set_xticks(x_ticks)
        axs[loc].set_yticks(np.arange(0.5, 1, 0.1))
        axs[loc].set_title(title_list[i//4])
        axs[loc].grid(True)
        axs[loc].set_xlabel(x_label)
        axs[loc].set_ylabel('Accuracy')
        axs[loc].legend(label_list[int(i%4/2)])
    fig.suptitle(main_title)
    plt.subplots_adjust(wspace=0.3, hspace=0.43)
    fig.savefig('examples/' + main_title + '.png')
    plt.show()
    
plotter(x=disp_imp, y=accuracy, x_ticks=[0.5, 1, 1.5], x_label='DI', main_title='accuracy vs DI')
plotter(x=fnr, y=accuracy, x_ticks=[-0.4, 0, 0.4], x_label='FNR', main_title='accuracy vs FNR')
plotter(x=fpr, y=accuracy, x_ticks=[-0.4, 0, 0.4], x_label='FPR', main_title='accuracy vs FPR')
