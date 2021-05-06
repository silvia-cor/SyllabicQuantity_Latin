from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from general.utils import pickled_resource
from SVM_classification import SVM_classification
from NN_classification import NN_classification
from NN_classification_attn import NN_classification_attn
from NN_classification_ensemble import NN_classification_ensemble
from dataset_prep.KabalaCorpusA_prep import dataset_KabalaCorpusA
from dataset_prep.LatinitasAntiqua_prep import dataset_LatinitasAntiqua
from dataset_prep.MedLatin_prep import dataset_MedLatin
from dataset_prep.all_prep import dataset_all

if __name__ == '__main__':
    n_sent = 10
    dataset_name = 'LatinitasAntiqua'
    data_path = f"../pickles/{dataset_name}/dataset_{dataset_name}_{n_sent}sent.pickle"
    svm_kfold_path = f'../pickles/{dataset_name}/svm_kfold_exp_{n_sent}.pickle'
    svm_loo_path = f'../pickles/{dataset_name}/svm_loo_exp_{n_sent}sent.pickle'
    nn_path = f'../pickles/{dataset_name}/nn_exp_{n_sent}sent.pickle'

    if dataset_name == 'KabalaCorpusA':
        dataset = pickled_resource(data_path, dataset_KabalaCorpusA, n_sent=n_sent)
    elif dataset_name == 'LatinitasAntiqua':
        dataset = pickled_resource(data_path, dataset_LatinitasAntiqua, n_sent=n_sent)
    elif dataset_name == 'MedLatin':
        dataset = pickled_resource(data_path, dataset_MedLatin, n_sent=n_sent)
    else:
        dataset = pickled_resource(data_path, dataset_all, n_sent=n_sent)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # loo = LeaveOneGroupOut()

    features_params = {'feature_selection_ratio': 1,
                       'function_words_freq': True,
                       'word_lengths_freq': True,
                       'sentence_lengths_freq': True,
                       'DVMA': False,
                       'DVSA': False,
                       'DVEX': False,
                       'DVL2': False,
                       'SQ': True,
                       'SQ_ngrams': [3, 3]}
    #SVM_classification(dataset, features_params, kfold, svm_kfold_path)

    NN_params = {'DVMA': True,
                 'DVSA': False,
                 'DVEX': False,
                 'DVL2': False,
                 'FAKE': False,
                 'SQ': True}
    NN_classification_ensemble(dataset, NN_params, dataset_name, n_sent, nn_path)
