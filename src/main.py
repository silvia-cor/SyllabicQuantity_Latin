from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from general.utils import pickled_resource
from SVM_classification import SVM_classification
from NN_classification import NN_classification
from dataset_prep.KabalaCorpusA_prep import dataset_KabalaCorpusA
from dataset_prep.LatinitasAntiqua_prep import dataset_LatinitasAntiqua
from dataset_prep.MedLatin_prep import dataset_MedLatin
from dataset_prep.all_prep import dataset_all

if __name__ == '__main__':
    n_sent = 10  # the documents will be divided into n sentences
    dataset_name = 'LatinitasAntiqua'  # dataset to be used (LatinitasAntiqua, KabalaCorpusA, MedLatin)
    nn_method = 'cnn_deep_ensemble'  # method for NN architecture (ccn_deep_ensemble, cnn_shallow_ensemble, cnn_cat, attn, lstm)
    cv_method = 'TrValTe'  # method for crossvalidation in SVM experiments (TrValTe, kfold, loo)

    # paths to store the pickles
    data_path = f"../pickles/{dataset_name}/dataset_{dataset_name}_{n_sent}sent.pickle"
    nn_path = f'../pickles/{dataset_name}/{nn_method}_exp_{n_sent}sent.pickle'
    svm_path = f'../pickles/{dataset_name}/svm_{cv_method}_exp_{n_sent}sent.pickle'

    # the dataset is loaded based on the name given
    if dataset_name == 'KabalaCorpusA':
        dataset = pickled_resource(data_path, dataset_KabalaCorpusA, n_sent=n_sent)
    elif dataset_name == 'LatinitasAntiqua':
        dataset = pickled_resource(data_path, dataset_LatinitasAntiqua, n_sent=n_sent)
    elif dataset_name == 'MedLatin':
        dataset = pickled_resource(data_path, dataset_MedLatin, n_sent=n_sent)
    else:
        dataset = pickled_resource(data_path, dataset_all, n_sent=n_sent)

    # parameters for SVM experiments
    features_params = {'function_words_freq': True,
                       'word_lengths_freq': True,
                       'sentence_lengths_freq': True,
                       'DVMA': True,
                       'DVSA': True,
                       'DVEX': True,
                       'DVL2': True,
                       'SQ': True,
                       'SQ_ngrams': [3, 7],
                       'SQ_selection_ratio': 0  # proportion of SQ features to be taken: 1: all, float: proportion, 0: optimized (only for TrValTe)
                       }
    SVM_classification(dataset, features_params, svm_path, cv_method=cv_method)

    # parameters for NN experiments
    NN_params = {'DVMA': False,
                 'DVSA': False,
                 'DVEX': False,
                 'DVL2': False,
                 'FAKE': False,
                 'SQ': True
                 }
    #NN_classification(dataset, NN_params, nn_method, dataset_name, n_sent, nn_path, batch_size=64)
