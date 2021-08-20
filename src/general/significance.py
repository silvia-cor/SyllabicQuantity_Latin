import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon


# convert the predictions into binary values (1 = correct author ; 0 = wrong author)
def _convert_preds(y_true, y_pred):
    result = []
    for true,pred in zip(y_true, y_pred):
        res = 1 if true == pred else 0
        result.append(res)
    return np.array(result)


# prepare the contingency table for the mcnemar's test
def _prepare_mcnemar_table(a,b):
    y_y= sum((a == 1) * (b == 1))
    y_n= sum((a == 1) * (b == 0))
    n_y= sum((a == 0) * (b == 1))
    n_n= sum((a == 0) * (b == 0))
    table = [[y_y,y_n],[n_y,n_n]]
    return table


def significance_test(y_true, y_baseline, y_method, baseline_name):
    # print('----- WILCOXON TEST AGAINST BASELINE -----')
    print(f'----- MCNEMAR TEST AGAINST {baseline_name} -----')
    y_baseline_conv = _convert_preds(y_true, y_baseline)
    y_method_conv = _convert_preds(y_true, y_method)
    # stat, p = wilcoxon(y_baseline_conv, y_method_conv, 'wilcox')
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
    return p










