import feature_extractor
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import sys
import numpy as np
import itertools

#prepares dataset for loo (transform everything into numpy array)
def __data_for_loo(dataset):
    authors = np.array(dataset.authors)
    titles_list = np.array(dataset.titles_list)
    data = np.array(list(itertools.chain.from_iterable(dataset.data)), dtype=object)
    authors_labels = np.array(list(itertools.chain.from_iterable(dataset.authors_labels)))
    titles_labels = np.array(list(itertools.chain.from_iterable(dataset.titles_labels)))
    return authors, titles_list, data, authors_labels, titles_labels

#prepares dataset for k-fold-cross-validation
#takes out first element in data and labels (which is the whole text) and transform in numpy array
def __data_for_kfold(dataset):
    authors = np.array(dataset.authors)
    titles_list = np.array(dataset.titles_list)
    data = np.array([sub_list[i] for sub_list in dataset.data for i in range(1, len(sub_list))])
    authors_labels = np.array([sub_list[i] for sub_list in dataset.authors_labels for i in range(1, len(sub_list))])
    titles_labels = np.array([sub_list[i] for sub_list in dataset.titles_labels for i in range(1, len(sub_list))])
    return authors, titles_list, data, authors_labels, titles_labels


#perform leave-one-out cross-validation
# training on fragments and whole texts, test only on whole texts
def loo_crossval(dataset, features_params):
    loo_fold = LeaveOneGroupOut()
    y_preds = []  # series of predictions
    y_tests = []  # series of true labels

    authors, titles_list, data, authors_labels, titles_labels = __data_for_loo(dataset)
    print('Tot. fragments + whole texts:', len(data))

    # loo cross-validation
    for i, (train_index, test_index) in enumerate(loo_fold.split(data, authors_labels, titles_labels)):
        # we are taking only test_index[0], which is the whole text
        print(f'----- LOO FOR {titles_list[titles_labels[test_index[0]]]} ({i+1}/{len(titles_list)}) -----')
        # cheking whether the text is the only one by that author
        if authors[authors_labels[test_index[0]]] not in authors[authors_labels[train_index]]:
            print("This is the only text by the author. Skipping.")
        else:
            x_train = data[train_index]
            x_test = [data[test_index[0]]]
            y_train = authors_labels[train_index]
            y_test = [authors_labels[test_index[0]]]

            print('FEATURE EXTRACTION')
            features = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test, **features_params)
            X_train, X_test = features.X_train, features.X_test

            print('CLASSIFICATION')
            # sometimes the learning method does not converge; this is to suppress a lot of warnings
            if not sys.warnoptions:
                import os, warnings
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')
            param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            #param_grid = {'C': [1, 10]}
            cls = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, n_jobs=4)
            # cls = GridSearchCV(SVC(kernel='linear', random_state=42), param_grid, n_jobs=4)
            cls.fit(X_train, y_train)
            print('Best C:', cls.best_params_['C'])
            y_pred = cls.predict(X_test)
            print('Author predicted:', authors[y_pred])
            y_preds.extend(y_pred)
            y_tests.extend(y_test)

    print('----- FINAL SCORE -----')
    macro_f1 = f1_score(y_tests, y_preds, average='macro')
    micro_f1 = f1_score(y_tests, y_preds, average='micro')
    print(f'Macro-F1: {macro_f1:.3f}')
    print(f'Micro-F1: {micro_f1:.3f}')


#perform k-fold cross-validation
#training and test only on fragments
def kfold_crossval(dataset, features_params, n_splits=5):
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_preds = []  # series of predictions
    y_tests = []  # series of true labels

    authors, titles_list, data, authors_labels, titles_labels = __data_for_kfold(dataset)
    print('Tot. fragments:', len(data))

    # k-fold cross-validation
    for i, (train_index, test_index) in enumerate(k_fold.split(data, authors_labels)):
        x_train = data[train_index]
        x_test = data[test_index]
        y_train = authors_labels[train_index]
        y_test = authors_labels[test_index]

        print(f'----- K-FOLD EXPERIMENT {i+1} -----')
        print('FEATURE EXTRACTION')
        features = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test, **features_params)
        X_train, X_test = features.X_train, features.X_test

        print('CLASSIFICATION')
        # sometimes the learning method does not converge; this is to suppress a lot of warnings
        if not sys.warnoptions:
            import os, warnings
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')
        param_grid= {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        #param_grid = {'C': [1, 10]}
        cls = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, n_jobs=4)
        #cls = GridSearchCV(SVC(kernel='linear', random_state=42), param_grid, n_jobs=4)
        cls.fit(X_train, y_train)
        print('Best C:', cls.best_params_['C'])
        y_pred = cls.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'F1: {f1:.3f}')
        y_preds.extend(y_pred)
        y_tests.extend(y_test)

    print('----- FINAL SCORE -----')
    macro_f1 = f1_score(y_tests, y_preds, average='macro')
    micro_f1 = f1_score(y_tests, y_preds, average='micro')
    print(f'Macro-F1: {macro_f1:.3f}')
    print(f'Micro-F1: {micro_f1:.3f}')