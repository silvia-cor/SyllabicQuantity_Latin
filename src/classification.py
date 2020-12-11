import feature_extractor
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import sys


#perform k-fold cross-validation only on fragments

def kfold_crossval(dataset, features_params, n_splits=5):
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_preds = []  # series of predictions
    y_tests = []  # series of true labels

    # k-fold cross validation
    for i, (train_index, test_index) in enumerate(k_fold.split(dataset.data, dataset.authors_labels)):
        x_train = dataset.data[train_index]
        x_test = dataset.data[test_index]
        y_train = dataset.authors_labels[train_index]
        y_test = dataset.authors_labels[test_index]

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