from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import sys
import feature_extractor

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')

# ------------------------------------------------------------------------
# class to do (classical ML) classification
# ------------------------------------------------------------------------

class Classifier:
    def __init__(self, dataset, features_params, cv='kfold', n_splits=5):

        self.dataset = dataset
        self.features_params = features_params
        self.y_tests = []
        self.y_preds = []
        assert cv in ['kfold', 'loo'], 'CV method must be either kfold or loo'

        if cv == 'kfold':
            self._kfold_crossval(n_splits)
        elif cv == 'loo':
            self._loo_crossval()
        else: pass

    # perform k-fold cross-validation
    # training and test only on fragments
    def _kfold_crossval(self, n_splits=5):
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        authors, titles_list, data, authors_labels, titles_labels = self.dataset.data_for_kfold()
        print('Tot. fragments:', len(data))

        # k-fold cross-validation
        for i, (train_index, test_index) in enumerate(k_fold.split(data, authors_labels)):
            x_train = data[train_index]
            x_test = data[test_index]
            y_train = authors_labels[train_index]
            y_test = authors_labels[test_index]

            print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
            print('FEATURE EXTRACTION')
            features = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test, **self.features_params)
            X_train, X_test = features.X_train, features.X_test

            print('CLASSIFICATION')
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # param_grid = {'C': [1, 10]}
            cls = GridSearchCV(LinearSVC(class_weight='balanced', random_state=42),
                               param_grid, scoring=make_scorer(f1_score, average='macro'), n_jobs=7)
            cls.fit(X_train, y_train)
            print('Best C:', cls.best_params_['C'])
            y_pred = cls.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f'F1: {f1:.3f}')
            self.y_preds.extend(y_pred)
            self.y_tests.extend(y_test)

        print('----- FINAL SCORE -----')
        macro_f1 = f1_score(self.y_tests, self.y_preds, average='macro')
        micro_f1 = f1_score(self.y_tests, self.y_preds, average='micro')
        print(f'Macro-F1: {macro_f1:.3f}')
        print(f'Micro-F1: {micro_f1:.3f}')



    # perform leave-one-out cross-validation
    # training on fragments and whole texts, test only on whole texts
    def _loo_crossval(self):
        loo_fold = LeaveOneGroupOut()
        authors, titles_list, data, authors_labels, titles_labels = self.dataset.data_for_loo()
        print('Tot. fragments + whole texts:', len(data))

        # loo cross-validation
        for i, (train_index, test_index) in enumerate(loo_fold.split(data, authors_labels, titles_labels)):
            # we are taking only test_index[0], which is the whole text
            print(f'----- LOO FOR {titles_list[titles_labels[test_index[0]]]} ({i + 1}/{len(titles_list)}) -----')
            # cheking whether the text is the only one by that author
            if authors[authors_labels[test_index[0]]] not in authors[authors_labels[train_index]]:
                print("This is the only text by the author. Skipping.")
            else:
                x_train = data[train_index]
                x_test = [data[test_index[0]]]
                y_train = authors_labels[train_index]
                y_test = [authors_labels[test_index[0]]]

                print('FEATURE EXTRACTION')
                features = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test, **self.features_params)
                X_train, X_test = features.X_train, features.X_test

                print('CLASSIFICATION')
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                # param_grid = {'C': [1, 10]}
                cls = GridSearchCV(LinearSVC(class_weight='balanced', random_state=42),
                                   param_grid, scoring=make_scorer(f1_score, average='macro'), n_jobs=7)
                cls.fit(X_train, y_train)
                print('Best C:', cls.best_params_['C'])
                y_pred = cls.predict(X_test)
                print('Author predicted:', authors[y_pred])
                self.y_preds.extend(y_pred)
                self.y_tests.extend(y_test)

        print('----- FINAL SCORE -----')
        macro_f1 = f1_score(self.y_tests, self.y_preds, average='macro')
        micro_f1 = f1_score(self.y_tests, self.y_preds, average='micro')
        print(f'Macro-F1: {macro_f1:.3f}')
        print(f'Micro-F1: {micro_f1:.3f}')

