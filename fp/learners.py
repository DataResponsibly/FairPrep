from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from aif360.algorithms.inprocessing import AdversarialDebiasing as AdversarialDebiasing360
# from aif360.algorithms.inprocessing import MetaFairClassifier
# from aif360.algorithms.inprocessing import PrejudiceRemover as PrejudiceRemover360

class Learner:

    def needs_annotated_data_for_prediction(self):
        return False


class LogisticRegression(Learner):

    def fit_model(self, annotated_train_data, fixed_random_seed):
        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.00005, 0.0001, 0.005, 0.001]
        }

        pipeline = Pipeline([
            ('learner', SGDClassifier(max_iter=1000, random_state=fixed_random_seed))
        ])

        # CAUTION: GridSearchCV parameter (fit_params) has been deprecated from
        #          the __init__ (constructor) in sk-learn version>=0.21 and has
        #          been introduced as a **kwargs in GridSearchCV.fit() method.

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

        fit_params = {'learner__sample_weight': annotated_train_data.instance_weights}        

        model = search.fit(annotated_train_data.features, annotated_train_data.labels, **fit_params)

        return model

    def name(self):
        return 'LogisticRegression'


class DecisionTree(Learner):

    def fit_model(self, annotated_train_data, fixed_random_seed):
        param_grid = {
            'learner__min_samples_split': range(20, 500, 10),
            'learner__max_depth': range(15, 30, 2),
            'learner__min_samples_leaf': [3, 4, 5, 10],
            "learner__criterion": ["gini", "entropy"]
        }

        pipeline = Pipeline([
            ('learner', DecisionTreeClassifier(random_state=fixed_random_seed))
        ])

        # CAUTION: GridSearchCV parameter (fit_params) has been deprecated from
        #          the __init__ (constructor) in sk-learn version>=0.21 and has
        #          been introduced as a **kwargs in GridSearchCV.fit() method.

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

        fit_params = {'learner__sample_weight': annotated_train_data.instance_weights}

        model = search.fit(annotated_train_data.features, annotated_train_data.labels, **fit_params)
        return model

    def name(self):
        return 'DecisionTree'



class NonTunedLogisticRegression(Learner):

    def fit_model(self, annotated_train_data, fixed_random_seed):
        return SGDClassifier(loss='log', random_state=fixed_random_seed)\
            .fit(annotated_train_data.features, annotated_train_data.labels,
                 sample_weight=annotated_train_data.instance_weights)

    def name(self):
        return 'LogisticRegression-notuning'



class NonTunedDecisionTree(Learner):

    def fit_model(self, annotated_train_data, fixed_random_seed):
        return DecisionTreeClassifier(random_state=fixed_random_seed)\
            .fit(annotated_train_data.features, annotated_train_data.labels,
                 sample_weight=annotated_train_data.instance_weights)

    def name(self):
        return 'DecisionTree-notuning'


# # TODO: Allow to pass parameters here (e.g., eta)
# class PrejudiceRemover(Learner):
#
#     def fit_model(self, annotated_train_data, fixed_random_seed):
#         return PrejudiceRemover360().fit(annotated_train_data)
#
#     def needs_annotated_data_for_prediction(self):
#         return True
#
#     def name(self):
#         return 'PrejudiceRemover'
#
#
#
# # TODO: this thing is too slow....
# # TODO: Allow to pass parameters here (e.g., tau, metric to optimize)
# class MetaFair(Learner):
#
#     def fit_model(self, annotated_train_data, fixed_random_seed):
#         return MetaFairClassifier().fit(annotated_train_data)
#
#     def name(self):
#         return 'MetaFair'


class AdversarialDebiasing(Learner):

    def __init__(self, session, privileged_groups, unprivileged_groups):
        self.session = session
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

    def fit_model(self, annotated_train_data, fixed_random_seed):
        debiased_model = AdversarialDebiasing360(privileged_groups=self.privileged_groups,
                                                 unprivileged_groups=self.unprivileged_groups,
                                                 scope_name='debiased_classifier',
                                                 debias=True,
                                                 sess=self.session,
                                                 seed=fixed_random_seed)
        debiased_model.fit(annotated_train_data)
        return debiased_model

    def name(self):
        return 'AdversarialDebiasing'

    def needs_annotated_data_for_prediction(self):
        return True
