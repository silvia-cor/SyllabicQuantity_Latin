import classification
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

def _convert_preds(y_true, y_pred):
    result = []
    for true,pred in zip(y_true, y_pred):
        res = 1 if true == pred else 0
        result.append(res)
    return np.array(result)

def _prepare_mcnemar_table(a,b):
    y_y= sum((a == 1) * (b ==1))
    y_n= sum((a == 1) * (b ==0))
    n_y= sum((a == 0) * (b ==1))
    n_n= sum((a == 0) * (b ==0))
    table = [[y_y,y_n],[n_y,n_n]]
    return table

#creates output file and write the true labels and the baseline predictions
def exp_create_base(dataset, file_path, features_params, CV, n_splits=5):
    classifier = classification.Classifier(dataset, features_params, CV, n_splits)
    df = {'True': classifier.y_tests, 'Baseline': classifier.y_preds}
    df_pd = pd.DataFrame(df)
    try:
        df_pd.to_pickle(file_path)
    except:
        print('Could not create the file :(')

def exp_significance_test(dataset, file_path, features_params, CV, n_splits=5):
    df = pd.read_pickle(file_path)
    y_true = df['True']
    y_baseline = df['Baseline']
    method_name = ''
    if features_params['char_ngrams'] == True:
        method_name += 'Char' + str(features_params['char_ngrams_range'])
    if features_params['syll_ngrams'] == True:
        method_name += 'Syll' + str(features_params['syll_ngrams_range'])
    method_name += ' FS(' + str(features_params['feature_selection_ratio']) + ')'

    if method_name not in df:
        print(f'----- EXPERIMENT {method_name} -----')
        classifier = classification.Classifier(dataset, features_params, CV, n_splits)
        df[method_name] = classifier.y_preds
        df_pd = pd.DataFrame(df)
        try:
            df_pd.to_pickle(file_path)
        except:
            print('Could not create the file :(')
    else:
        print(f'Experiment {method_name} already done!')

    y_method = df[method_name]

    print('----- F1 SCORE -----')
    macro_f1 = f1_score(y_true, y_method, average='macro')
    micro_f1 = f1_score(y_true, y_method, average='micro')
    print(f'Macro-F1: {macro_f1:.3f}')
    print(f'Micro-F1: {micro_f1:.3f}')

    #print('----- WILCOXON TEST AGAINST BASELINE -----')
    print('----- MCNEMAR TEST AGAINST BASELINE -----')
    y_baseline_conv = _convert_preds(y_true, y_baseline)
    y_method_conv = _convert_preds(y_true, y_method)
    #stat, p = wilcoxon(y_baseline_conv, y_method_conv, 'wilcox')
    test_table = _prepare_mcnemar_table(y_baseline_conv, y_method_conv)
    test_result = mcnemar(test_table)
    stat, p = test_result.statistic, test_result.pvalue
    print(f'Statistics= {stat:.3f}')
    print(f'p= {p:.3f}')
    alpha = 0.05
    if p > alpha:
        print('Same proportion (difference is not significant)')
    else:
        print('Different proportion (difference is significant)')










