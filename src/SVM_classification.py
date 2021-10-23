from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split, PredefinedSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score
import sys
import pickle
import numpy as np
from feature_extractor import featuresExtractor
from general.helpers import data_for_kfold, data_for_loo
from general.significance import significance_test
from tqdm.contrib.concurrent import process_map

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


# ------------------------------------------------------------------------
# class to do (classical ML) classification
# ------------------------------------------------------------------------


def SVM_classification(dataset, features_params, pickle_path, cv_method='1fold'):
    assert cv_method in ['TrValTe', 'kfold', 'loo'], 'CV method must be either TrValTe, kfold or loo'
    assert features_params[
               'SQ_selection_ratio'] != 0 or cv_method == 'TrValTe', 'SQ selection optimization only for TrValTe method'
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
        if cv_method == 'kfold':
            y_all_pred, y_all_te, best_Cs, tot_features = _kfold_crossval(dataset, features_params)
        elif cv_method == 'loo':
            y_all_pred, y_all_te, best_Cs, tot_features = _loo_crossval(dataset, features_params)
        else:
            y_all_pred, y_all_te, best_Cs, tot_features = _TrValTe_crossval(dataset, features_params)
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
    print(f'Micro-F1: {df[method_name]["microF1"]:.3f}')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(df, handle)
    # significance test if SQ are in the features with another method
    # significance test is against the same method without SQ
    if ' + SQ' in method_name:
        baseline = method_name.split(' + SQ')[0]
        if baseline in df:
            print(f'COMPARISON WITH BASELINE {baseline}')
            delta_macro = (df[method_name]['macroF1'] - df[baseline]['macroF1']) / df[baseline]['macroF1'] * 100
            delta_micro = (df[method_name]['microF1'] - df[baseline]['microF1']) / df[baseline]['microF1'] * 100
            print(f'Macro-F1 Delta %: {delta_macro:.2f}')
            print(f'Micro-F1 Delta %: {delta_micro:.2f}')
            significance_test(df['True']['labels'], df[baseline]['preds'], df[method_name]['preds'], baseline)
        else:
            print(f'No {baseline} saved, significance test cannot be performed :/')
    else:
        print('No significance test requested')


# generates the name of the method used to save the results
def _create_method_name(features_params):
    methods = []
    dv_methods = ['DVMA', 'DVSA', 'DVEX', 'DVL2']
    method_name = ''
    if features_params['function_words_freq'] and features_params['word_lengths_freq'] and features_params[
        'sentence_lengths_freq']:
        method_name += 'BaseFeatures'
    for method in dv_methods:
        if features_params[method]:
            methods.append(method)
    if len(methods) == 4:
        if method_name != '':
            method_name += ' + '
        method_name += 'ALLDV'
    elif len(methods) > 0:
        if method_name != '':
            method_name += ' + '
        method_name += ' + '.join(methods)
    if features_params['SQ']:
        if method_name != '':
            method_name += ' + '
        method_name += 'SQ' + '[' + str(features_params['SQ_ngrams'][0]) + ',' + str(
            features_params['SQ_ngrams'][1]) + ']'
        if features_params['SQ_selection_ratio'] != 1:
            method_name += '|FS'
            if features_params['SQ_selection_ratio'] == 0:
                method_name += '(opt)'
            else:
                method_name += '(' + str(features_params['SQ_selection_ratio']) + ')'
    return method_name


# perform kfold cross-validation using a LinearSVM
# training and testing only on fragments
# optimization done only for SVM parameter C via GridSearch
def _kfold_crossval(dataset, features_params):
    authors, titles, data, data_cltk, authors_labels, titles_labels = data_for_kfold(dataset)
    print('Tot. fragments:', len(data))
    y_all_pred = []
    y_all_te = []
    best_Cs = []
    tot_features = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kfold.split(data, authors_labels)):
        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        x_tr = data[train_index]
        x_tr_cltk = data_cltk[train_index]
        x_te = data[test_index]
        x_te_cltk = data_cltk[test_index]
        y_tr = authors_labels[train_index]
        y_te = authors_labels[test_index]
        print('FEATURE EXTRACTION')
        X_tr, X_te = featuresExtractor(x_tr, x_te, x_tr_cltk, x_te_cltk, y_tr, **features_params)
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


def __single_TrVal_exp(configuration):
    C = configuration[0]
    feat_params = configuration[1]
    dataset = configuration[2]
    cls = LinearSVC(class_weight='balanced', random_state=42, C=C)
    X_tr, X_val = featuresExtractor(dataset['x_tr'], dataset['x_val'], dataset['x_tr_cltk'], dataset['x_val_cltk'],
                                    dataset['y_tr'], **feat_params)
    cls.fit(X_tr, dataset['y_tr'])
    y_pred = cls.predict(X_val)
    f1 = f1_score(dataset['y_val'], y_pred, average='macro')
    return f1


# perform train-val-test validation using a LinearSVM
# training and testing only on fragments
# optimization of SVM parameter C (and SQ_selection_ratio if required) via GridSearch from scratch
def _TrValTe_crossval(dataset, features_params):
    authors, titles, data, data_cltk, authors_labels, titles_labels = data_for_kfold(dataset)
    # divide the dataset into train+val and test
    x_trval, x_te, x_trval_cltk, x_te_cltk, y_trval, y_te = train_test_split(data, data_cltk, authors_labels,
                                                                             test_size=0.1, random_state=42,
                                                                             stratify=authors_labels)
    # divide the train+val so that the dataset is train/val/test
    x_tr, x_val, x_tr_cltk, x_val_cltk, y_tr, y_val = train_test_split(x_trval, x_trval_cltk, y_trval, test_size=0.1,
                                                                       random_state=42, stratify=y_trval)
    exp_dataset = {'x_tr': x_tr, 'x_val': x_val, 'x_tr_cltk': x_tr_cltk, 'x_val_cltk': x_val_cltk, 'y_tr': y_tr,
                   'y_val': y_val}
    print(f'#training samples = {len(y_tr)}')
    print(f'#validation samples = {len(y_val)}')
    print(f'#test samples = {len(y_te)}')
    fs_ratios = [1, 0.1, 0.2, 0.3, 0.4, 0.5]
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    configurations = []
    for C in Cs:
        if features_params['SQ'] and features_params['SQ_selection_ratio'] == 0:
            for fs_ratio in fs_ratios:
                new_feat_params = features_params.copy()
                new_feat_params['SQ_selection_ratio'] = fs_ratio
                configurations.append((C, new_feat_params, exp_dataset))
        else:
            configurations.append((C, features_params, exp_dataset))
    print('PARAMETERS OPTIMIZATION')
    results = process_map(__single_TrVal_exp, configurations, max_workers=8)
    best_result_idx = results.index(max(results, key=lambda result: result))
    best_C = configurations[best_result_idx][0]
    best_feat_params = configurations[best_result_idx][1]
    print('BEST MODEL')
    print('Best C:', best_C)
    if features_params['SQ'] and features_params['SQ_selection_ratio'] == 0:
        print('Best f1_ratio:', best_feat_params['SQ_selection_ratio'])
    print(f'Best macro-f1: {results[best_result_idx]:.3f}')
    print('CLASSIFICATION')
    cls = LinearSVC(class_weight='balanced', random_state=42, C=best_C)
    X_trval, X_te = featuresExtractor(x_trval, x_te, x_trval_cltk, x_te_cltk, y_trval, **best_feat_params)
    cls.fit(X_trval, y_trval)
    y_pred = cls.predict(X_te)
    print("Training shape: ", X_trval.shape)
    print("Test shape: ", X_te.shape)
    return y_pred, y_te, best_C, X_te.shape[1]


# perform leave-one-out cross-validation
# training on fragments and whole texts, test only on whole texts
def _loo_crossval(dataset, features_params):
    authors, titles, data, data_cltk, authors_labels, titles_labels = data_for_loo(dataset)
    print('Tot. fragments + whole texts:', len(data))
    y_all_pred = []
    y_all_te = []
    best_Cs = []
    tot_features = []
    loo = LeaveOneGroupOut()
    # loo cross-validation
    for i, (train_index, test_index) in enumerate(loo.split(data, authors_labels, titles_labels)):
        # we are taking only test_index[0], which is the whole text
        print(f'----- LOO FOR {titles[titles_labels[test_index[0]]]} ({i + 1}/{len(titles)}) -----')
        # cheking whether the text is the only one by that author
        if authors[authors_labels[test_index[0]]] not in authors[authors_labels[train_index]]:
            print("This is the only text by the author. Skipping.")
        else:
            x_tr = data[train_index]
            x_tr_cltk = data_cltk[train_index]
            x_te = [data[test_index[0]]]
            x_te_cltk = [data_cltk[test_index[0]]]
            y_tr = authors_labels[train_index]
            y_te = [authors_labels[test_index[0]]]
            print('FEATURE EXTRACTION')
            X_tr, X_te = featuresExtractor(x_tr, x_te, x_tr_cltk, x_te_cltk, y_tr, **features_params)
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
            print('Author predicted:', authors[y_pred])
            y_all_pred.extend(y_pred)
            y_all_te.extend(y_te)
    return y_all_pred, y_all_te, best_Cs, tot_features
