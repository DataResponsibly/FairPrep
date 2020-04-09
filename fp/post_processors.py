from fp.fixed.reject_option_classification import RejectOptionClassification
from fp.fixed.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing


class NoPostProcessing:

    def post_process(self, validation_dataset, validation_dataset_with_predictions, testset_with_predictions, seed,
                     privileged_groups, unprivileged_groups):
        return testset_with_predictions

    def name(self):
        return 'no_post_processing'


class RejectOptionPostProcessing:

    def post_process(self, validation_dataset, validation_dataset_with_predictions, testset_with_predictions, seed,
                     privileged_groups, unprivileged_groups):

        roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        roc = roc.fit(validation_dataset, validation_dataset_with_predictions)

        return roc.predict(testset_with_predictions)

    def name(self):
        return 'reject_option'


class EqualOddsPostProcessing:

    def post_process(self, validation_dataset, validation_dataset_with_predictions, testset_with_predictions, seed,
                     privileged_groups, unprivileged_groups):

        post_processor = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups, seed=seed)

        post_processor = post_processor.fit(validation_dataset, validation_dataset_with_predictions)
        return post_processor.predict(testset_with_predictions)

    def name(self):
        return 'eq_odds'


class CalibratedEqualOddsPostProcessing:

    def post_process(self, validation_dataset, validation_dataset_with_predictions, testset_with_predictions, seed,
                     privileged_groups, unprivileged_groups):

        post_processor = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                                        unprivileged_groups=unprivileged_groups, seed=seed)

        post_processor = post_processor.fit(validation_dataset, validation_dataset_with_predictions)
        return post_processor.predict(testset_with_predictions)

    def name(self):
        return 'calibrated_eq_odds'
