from pathlib import Path
import unittest

import warnings
warnings.filterwarnings("ignore")
from FairPrep.stage import *


class TestStringMethods(unittest.TestCase):
    def test_data(self):
        data_dir = Path(__file__).parent.parent / 'data'
        input_data = data_dir / 'german_AIF_test.csv'

        assert data_dir.exists()
        assert input_data.exists()

    def test_pipeline(self):

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
                                           sensitive_attributes=sensi_atts, protected_groups=protected_groups,
                                           special_value_mapping=value_mapping,
                                           debias_attribute=debias_focus_att, numerical_atts=numerical_atts,
                                           categorical_atts=categorical_atts,
                                           seed=global_seed, verbose=True)

        # set each stage using above stage objects
        fairprep_builder.setPreprocessStage(preprocess_stage_builder=preprocess_stage)
        fairprep_builder.setModelStage(model_stage_builder=model_stage)
        fairprep_builder.setPostprocessStage(postprocess_stage_builder=postprocess_stage)

        # run the pipeline that is built from the steps specified in above three stages
        train_after, val_after, test_after = fairprep_builder.runPipeline(save_interdata=True)

        output_log_path = fairprep_builder.engine.log_dir_name

        assert Path(output_log_path).exists()
        assert Path(output_log_path + 'german_AIF_test.json').exists()
        assert Path(output_log_path + 'Metrics-0-test.csv').exists()


if __name__ == '__main__':
    unittest.main()


