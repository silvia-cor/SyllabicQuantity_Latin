from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from general.utils import pickled_resource
from SVM_classification import SVM_classification
from NN_classification import NN_classification
from general.visualization import f1_scatterplot
from dataset_prep.LatinitasAntiqua_prep import dataset_LatinitasAntiqua

if __name__ == '__main__':
    n_sent = 10
    data_path = f"../pickles/dataset_LatinitasAntiqua_{n_sent}sent.pickle"
    svm_kfold_path = f'../pickles/svm_kfold_exp_{n_sent}sent.pickle'
    svm_loo_path = f'../pickles/svm_loo_exp_{n_sent}sent.pickle'
    nn_path = f'../pickles/nn_exp_{n_sent}sent.pickle'

    dataset = pickled_resource(data_path, dataset_LatinitasAntiqua, n_sent=n_sent)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #loo = LeaveOneGroupOut()




    features_params = {'feature_selection_ratio': 0.3,
                       'function_words_freq': True,
                       'word_lengths_freq': True,
                       'sentence_lengths_freq': True,
                       'DVMA': False,
                       'DVSA': False,
                       'DVEX': True,
                       'DVL2': True,
                       'SQ': True,
                       'SQ_ngrams': [3,7]}
    #SVM_classification(dataset, features_params, kfold, svm_kfold_path)


    NN_params = {'DVMA': True,
                 'DVSA': False,
                 'DVEX': False,
                 'DVL2': False,
                 'FAKE': False,
                 'SQ': False}

    NN_classification(dataset, NN_params, kfold, n_sent, nn_path)
