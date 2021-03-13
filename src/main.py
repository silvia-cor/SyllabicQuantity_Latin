from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from general.utils import pickled_resource
from SVM_classification import SVM_classification
from general.visualization import f1_scatterplot
from dataset_prep.LatinitasAntiqua_prep import dataset_LatinitasAntiqua


n_sent = 100
data_path = f"../pickles/dataset_LatinitasAntiqua_{n_sent}sent.pickle"
svm_kfold_path = f'../pickles/svm_kfold_exp_{n_sent}sent.pickle'
svm_loo_path = f'../pickles/svm_loo_exp_{n_sent}sent.pickle'

dataset = pickled_resource(data_path, dataset_LatinitasAntiqua, n_sent=n_sent)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
loo = LeaveOneGroupOut()

features_params = {'feature_selection_ratio': 1,
                   'function_words_freq': True,
                   'word_lengths_freq': True,
                   'sentence_lengths_freq': True,
                   'DVMA':True,
                   'DVSA':False,
                   'DVEX':False,
                   'DVL2':False,
                   'SQ': False,
                   'SQ_ngrams': [3, 3]}

SVM_classification(dataset, features_params, kfold, svm_kfold_path)

features_params = {'feature_selection_ratio': 1,
                   'function_words_freq': True,
                   'word_lengths_freq': True,
                   'sentence_lengths_freq': True,
                   'DVMA':False,
                   'DVSA':True,
                   'DVEX':False,
                   'DVL2':False,
                   'SQ': False,
                   'SQ_ngrams': [3, 3]}

SVM_classification(dataset, features_params, kfold, svm_kfold_path)

features_params = {'feature_selection_ratio': 1,
                   'function_words_freq': True,
                   'word_lengths_freq': True,
                   'sentence_lengths_freq': True,
                   'DVMA':False,
                   'DVSA':False,
                   'DVEX':True,
                   'DVL2':False,
                   'SQ': False,
                   'SQ_ngrams': [3, 3]}

SVM_classification(dataset, features_params, kfold, svm_kfold_path)

features_params = {'feature_selection_ratio': 1,
                   'function_words_freq': True,
                   'word_lengths_freq': True,
                   'sentence_lengths_freq': True,
                   'DVMA':False,
                   'DVSA':False,
                   'DVEX':False,
                   'DVL2':True,
                   'SQ': False,
                   'SQ_ngrams': [3, 3]}

SVM_classification(dataset, features_params, kfold, svm_kfold_path)

feature_sels = [1]
mins = [3,4,5,6,7]
maxs = [3,4,5,6,7]
for feature_sel in feature_sels:
    for min in mins:
        for max in maxs:
            if (min > max):
                pass
            else:
                features_params = {'feature_selection_ratio': feature_sel,
                                   'function_words_freq': True,
                                   'word_lengths_freq': True,
                                   'sentence_lengths_freq': True,
                                   'DVMA': False,
                                   'DVSA': False,
                                   'DVEX': False,
                                   'DVL2': False,
                                   'SQ': True,
                                   'SQ_ngrams': [min, max]}

                SVM_classification(dataset, features_params, kfold, svm_kfold_path)

feature_sels = [0.1, 0.3]
for feature_sel in feature_sels:
    features_params = {'feature_selection_ratio': feature_sel,
                       'function_words_freq': True,
                       'word_lengths_freq': True,
                       'sentence_lengths_freq': True,
                       'DVMA': False,
                       'DVSA': False,
                       'DVEX': False,
                       'DVL2': False,
                       'SQ': True,
                       'SQ_ngrams': [3, 7]}
    SVM_classification(dataset, features_params, kfold, svm_kfold_path)

feature_sels = [0.1, 0.3]
ns = [8,9,10]
for feature_sel in feature_sels:
    for n in ns:
        features_params = {'feature_selection_ratio': feature_sel,
                           'function_words_freq': True,
                           'word_lengths_freq': True,
                           'sentence_lengths_freq': True,
                           'DVMA': False,
                           'DVSA': False,
                           'DVEX': False,
                           'DVL2': False,
                           'SQ': True,
                           'SQ_ngrams': [n, n]}
        SVM_classification(dataset, features_params, kfold, svm_kfold_path)