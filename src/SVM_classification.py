from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import sys
import pickle
import numpy as np
from feature_extractor import featuresExtractor
from general.helpers import data_for_kfold, data_for_loo
from general.significance import significance_test

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')

# ------------------------------------------------------------------------
# class to do (classical ML) classification
# ------------------------------------------------------------------------

def SVM_classification(dataset, features_params, cv_method, pickle_path):
    assert isinstance(cv_method, (StratifiedKFold, LeaveOneGroupOut)), 'CV method must be either kfold or loo'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            df = pickle.load(handle)
    else:
        df = {}

    method_name = _create_method_name(features_params)
    if method_name in df:
        print(f'Experiment {method_name} already done!')
    else:
        print(f'----- SVM EXPERIMENT {method_name} -----')
        if isinstance(cv_method, StratifiedKFold):
            y_all_pred, y_all_te, best_Cs, tot_features = _kfold_crossval(dataset, features_params, cv_method)
        else:
            y_all_pred, y_all_te, best_Cs, tot_features = _loo_crossval(dataset, features_params, cv_method)
        if 'True' not in df:
            df['True'] = {}
            df['True']['labels'] = y_all_te
        df[method_name] = {}
        df[method_name]['preds'] = y_all_pred
        df[method_name]['best_Cs'] = best_Cs
        df[method_name]['tot_features'] = tot_features
        macro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='macro')
        micro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='micro')
        df[method_name]['macroF1'] = macro_f1
        df[method_name]['microF1'] = micro_f1
    print('----- THE END -----')
    print('Tot. features (mean):', int(np.round(np.mean(df[method_name]['tot_features']))))
    print(f'Macro-F1: {df[method_name]["macroF1"]:.3f}')
    print(f'Micro-F1: {df[method_name]["microF1"] :.3f}')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(df, handle)

    #significance test if SQ are in the features with another method
    #significance test is against the same method without SQ
    if ' + SQ' in method_name:
        baseline = method_name.split(' + SQ')[0]
        if baseline in df:
            significance_test(df['True']['labels'], df[baseline]['preds'], df[method_name]['preds'], baseline)
        else:
            print(f'No {baseline} saved, significance test cannot be performed :/')
    else:
        print('No significance test requested')



#generates the name of the method used to save the results
def _create_method_name(features_params):
    methods = []
    dv_methods = ['DVMA', 'DVSA', 'DVEX', 'DVL2']
    method_name = ''
    for method in dv_methods:
        if features_params[method]:
            methods.append(method)
    if len(methods) == 4:
        method_name = 'ALLDV'
    else:
        method_name = ' + '.join(methods)
    if features_params['SQ']:
        if method_name != '':
            method_name += ' + '
        method_name += 'SQ' + '[' + str(features_params['SQ_ngrams'][0]) + ',' + str(features_params['SQ_ngrams'][1]) + ']'
        method_name += '|FS(' + str(features_params['feature_selection_ratio']) + ')'
    return method_name


#perform kfold cross-validation using a LinearSVM
#training and testing only on fragments
def _kfold_crossval(dataset, features_params, kfold):
    authors, titles, data, authors_labels, titles_labels = data_for_kfold(dataset)
    print('Tot. fragments:', len(data))
    y_all_pred = []
    y_all_te = []
    best_Cs = []
    tot_features = []

    for i, (train_index, test_index) in enumerate(kfold.split(data, authors_labels)):
        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        x_tr = data[train_index]
        x_te = data[test_index]
        y_tr = authors_labels[train_index]
        y_te = authors_labels[test_index]
        print('FEATURE EXTRACTION')
        X_tr, X_te = featuresExtractor(x_tr, x_te, y_tr, **features_params)
        print("Training shape: ", X_tr.shape)
        print("Test shape: ", X_te.shape)
        tot_features.append(X_tr.shape[1])

        print('CLASSIFICATION')
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        cls = GridSearchCV(LinearSVC(class_weight='balanced', random_state=42), param_grid,
            scoring=make_scorer(f1_score, average='macro'), n_jobs=7)
        cls.fit(X_tr, y_tr)
        best_C = cls.best_params_['C']
        print('Best C:', best_C)
        best_Cs.append(best_C)
        y_pred = cls.predict(X_te)
        f1 = f1_score(y_te, y_pred, average='macro')
        print(f'F1: {f1:.3f}')
        y_all_pred.extend(y_pred)
        y_all_te.extend(y_te)

    return y_all_pred, y_all_te, best_Cs, tot_features


# perform leave-one-out cross-validation
# training on fragments and whole texts, test only on whole texts
def _loo_crossval(dataset, features_params, loo):
    authors, titles, data, authors_labels, titles_labels = data_for_loo(dataset)
    print('Tot. fragments + whole texts:', len(data))
    y_all_pred = []
    y_all_te = []
    best_Cs = []
    tot_features = []

    # loo cross-validation
    for i, (train_index, test_index) in enumerate(loo.split(data, authors_labels, titles_labels)):
        # we are taking only test_index[0], which is the whole text
        print(f'----- LOO FOR {titles[titles_labels[test_index[0]]]} ({i + 1}/{len(titles)}) -----')
        # cheking whether the text is the only one by that author
        if authors[authors_labels[test_index[0]]] not in authors[authors_labels[train_index]]:
            print("This is the only text by the author. Skipping.")
        else:
            x_tr = data[train_index]
            x_te = [data[test_index[0]]]
            y_tr = authors_labels[train_index]
            y_te = [authors_labels[test_index[0]]]
            print('FEATURE EXTRACTION')
            X_tr, X_te = featuresExtractor(x_tr, x_te, y_tr, **features_params)
            print("Training shape: ", X_tr.shape)
            print("Test shape: ", X_te.shape)
            tot_features.append(X_tr.shape[1])

            print('CLASSIFICATION')
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # param_grid = {'C': [1, 10]}
            cls = GridSearchCV(LinearSVC(class_weight='balanced', random_state=42), param_grid,
                scoring=make_scorer(f1_score, average='macro'), n_jobs=7)
            cls.fit(X_tr, y_tr)
            best_C = cls.best_params_['C']
            print('Best C:', best_C)
            best_Cs.append(best_C)
            y_pred = cls.predict(X_te)
            print('Author predicted:', authors[y_pred])
            y_all_pred.extend(y_pred)
            y_all_te.extend(y_te)

    return y_all_pred, y_all_te, best_Cs, tot_features