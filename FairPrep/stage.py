"""
    Classes of the builder object, which are the interface classes of users' inputs.

"""


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
from FairPrep.pipeline import FairPipeline

class FairPrepBuilder():
    def __init__(self, data_file, target_col, positive_target_value, sensitive_attributes, protected_groups, special_value_mapping, debias_attribute, numerical_atts, categorical_atts, seed, data_sep=None, na_mark=None, verbose=True):

        self.data_file = data_file
        self.target_col = target_col
        self.positive_target_value = positive_target_value
        self.sensitive_attributes = sensitive_attributes
        self.protected_groups = protected_groups
        self.special_value_mapping = special_value_mapping

        self.debias_attribute = debias_attribute

        # for integrity check of the inputs of some steps
        self.numerical_atts = numerical_atts
        self.categorical_atts = categorical_atts

        self.seed = seed


        self.pipeline = []

        self.engine = FairPipeline(data_file, target_col, positive_target_value, sensitive_attributes, protected_groups, special_value_mapping, debias_attribute, seed, sep_flag=data_sep, na_mark=na_mark, verbose=verbose)

    def setPreprocessStage(self, preprocess_stage_builder):
        # TODO: integrity check for the input pipeline
        self.pipeline += preprocess_stage_builder.preprocess_pipeline


    def setModelStage(self, model_stage_builder):
        # TODO: integrity check for the input pipeline
        self.pipeline += model_stage_builder.model

    def setPostprocessStage(self, postprocess_stage_builder):
        # TODO: integrity check for the input pipeline
        self.pipeline += postprocess_stage_builder.postprocessor


    def getPipeline(self):
        return self.pipeline

    def runPipeline(self, save_interdata=True):
        return self.engine.run_pipeline(self.pipeline, save_interdata=save_interdata)




class PreprocessStageBuilder():
    def __init__(self):
        self.preprocess_pipeline = []

    def setSplitter(self, splitter):
        self.preprocess_pipeline.append(splitter)

    def setSampler(self, sampler):
        self.preprocess_pipeline.append(sampler)

    def setImputer(self, imputer):
        self.preprocess_pipeline.append(imputer)

    def setScaler(self, scaler):
        self.preprocess_pipeline.append(scaler)

    def setCategorizer(self, categotizer):
        self.preprocess_pipeline.append(categotizer)

    def setEncoder(self, encoder):
        self.preprocess_pipeline.append(encoder)

    def setMappingEncoder(self, mapping_encoder):
        self.preprocess_pipeline.append(mapping_encoder)

    def setFairPreprocessor(self, fair_preprocessor):
        self.preprocess_pipeline.append(fair_preprocessor)


class ModelStageBuilder():

    def __init__(self):
        self.model = []

    def setModel(self, model):
        self.model.append(model)

class PostprocessStageBuilder():
    def __init__(self):
        self.postprocessor = []

    def setPostprocessor(self, postprocessor):
        self.postprocessor.append(postprocessor)


if __name__ == '__main__':
    # specify the basic descriptive parameters of the input data
    data_file = "../data/german_AIF_test.csv"
    y_col = "credit"
    y_posi = ["good"]
    sensi_atts = ["age", "sex"]

    protected_groups = {"age": "young", "sex": "female"}
    value_mapping = {"female": 0, "male": 1, "good": 1, "bad": 0, "young": 0, "old": 1}

    debias_focus_att = "sex"
    global_seed = 0

    numerical_atts = ["month", "credit_amount"]
    categorical_atts = ["status", "employment", "housing"]

    # initialize each stage obejct in order to build a fair pipeline
    preprocess_stage = PreprocessStageBuilder()
    model_stage = ModelStageBuilder()
    postprocess_stage = PostprocessStageBuilder()

    # specify the method for each preprocessing step
    preprocess_stage.setSplitter(RandomSplitter([0.5, 0.3, 0.2], global_seed))
    preprocess_stage.setSampler(NoSampler())
    preprocess_stage.setImputer(NoImputer())
    preprocess_stage.setScaler(SK_MinMaxScaler(numerical_atts))
    preprocess_stage.setCategorizer(NoBinarizer())
    preprocess_stage.setEncoder(OneHotEncoder(categorical_atts))
    preprocess_stage.setMappingEncoder(MappingEncoder([y_col] + sensi_atts, value_mapping))
    preprocess_stage.setFairPreprocessor(AIF_Reweighing(y_col, debias_focus_att))

    # specify the method for the model step
    model_stage.setModel(OPT_LogisticRegression(y_col, global_seed))

    # specify the method for the postprocessor step
    postprocess_stage.setPostprocessor(NoFairPostprocessor())

    # initialize a fair prep object using the above data descriptive parameters
    fairprep_builder = FairPrepBuilder(data_file=data_file, target_col=y_col, positive_target_value=y_posi,
                               sensitive_attributes=sensi_atts, protected_groups=protected_groups, special_value_mapping=value_mapping,
                               debias_attribute=debias_focus_att, numerical_atts=numerical_atts, categorical_atts=categorical_atts,
                               seed=global_seed, verbose=True)


    # set each stage using above stage objects
    fairprep_builder.setPreprocessStage(preprocess_stage_builder=preprocess_stage)
    fairprep_builder.setModelStage(model_stage_builder=model_stage)
    fairprep_builder.setPostprocessStage(postprocess_stage_builder=postprocess_stage)

    # run the pipeline that is built from the steps specified in above three stages
    train_after, val_after, test_after = fairprep_builder.runPipeline(save_interdata=True)
    train_after.to_csv(fairprep_builder.engine.log_dir_name + "Final_train.csv", index=False)



    # TODO: add iteration for multiple inputs
    # for global_seed in [1,2,3,4]:
    #     samplers = [NoSampler(), RandomSampler(sample_n=800, seed=global_seed)]
    #
    #     splitters = [RandomSplitter([0.5, 0.3, 0.2], global_seed)]
    #     imputers = [NoImputer()]
    #     scalers = [SK_MinMaxScaler(numerical_atts)]
    #     categorizers = [NoBinarizer()]
    #     encoders = [OneHotEncoder(categorical_atts)]
    #     special_encoders = [MappingEncoder([y_col] + sensi_atts, value_mapping)]
    #     preprocessors = [AIF_Reweighing(y_col, debias_focus_att)]
    #
    #     models = [OPT_LogisticRegression(y_col, global_seed)]
    #
    #     postprocessors = [NoFairPostprocessor()]
    #
    #     for cur_stage in


