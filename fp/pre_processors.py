from aif360.algorithms.preprocessing import Reweighing as ReweighingAIF360
from aif360.algorithms.preprocessing import LFR as LFR360
from aif360.algorithms.preprocessing import DisparateImpactRemover


class NoPreProcessing:

    def pre_process(self, annotated_train_data, privileged_groups, unprivileged_groups):
        return annotated_train_data

    def name(self):
        return 'no_pre_processing'


class Reweighing:

    def pre_process(self, annotated_data, privileged_groups, unprivileged_groups):

        reweighing_transformer = ReweighingAIF360(unprivileged_groups, privileged_groups)
        return reweighing_transformer.fit(annotated_data).transform(annotated_data)

    def name(self):
        return 'reweighing'


class DIRemover:

    def __init__(self, repair_level):
        self.repair_level = repair_level


    def pre_process(self, annotated_data, privileged_groups, unprivileged_groups):
        disparate_impact_remover = DisparateImpactRemover(repair_level=self.repair_level)
        return disparate_impact_remover.fit_transform(annotated_data)

    def name(self):
        return 'diremover-' + str(self.repair_level)


class LFR:

    def __init__(self):
        self.fitted_preprocessor = None

    def pre_process(self, annotated_data, privileged_groups, unprivileged_groups):
        return LFR360(unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups)\
                    .fit_transform(annotated_data)

    def name(self):
        return 'LFR'
