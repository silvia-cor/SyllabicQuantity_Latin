from general.utils import pickled_resource
from SVM_classification import SVM_classification
from NN_classification import NN_classification
import importlib

if __name__ == '__main__':
    n_sent = 10  # the documents will be divided into n sentences
    dataset_name = 'MedLatin'  # dataset to be used (LatinitasAntiqua, KabalaCorpusA, MedLatin)
    learner = 'cnn_shallow_ensemble'  # learner to be used (svm, cnn_deep_ensemble)
    cv_method = '1fold'  # method for crossvalidation in SVM experiments (1fold, kfold, loo)
    assert learner in ['svm', 'cnn_shallow_ensemble'], \
        'Learner not implemented. Options: svm, cnn_deep_ensemble.'
    assert cv_method in ['1fold', 'kfold', 'loo'], 'CV method must be either 1fold, kfold or loo'

    # paths to store the pickles
    data_path = f"../pickles/{dataset_name}/{dataset_name}_{n_sent}sent.pickle"
    pickle_path = f'../pickles/{dataset_name}/{learner}_{cv_method}_{n_sent}sent.pickle'

    # the dataset is loaded based on the name given
    module = importlib.import_module(f'dataset_prep.{dataset_name}_prep')
    dataset = pickled_resource(data_path, getattr(module, f'dataset_{dataset_name}'), n_sent=n_sent)

    if learner == 'svm':
        # parameters for SVM experiments
        features_params = {'function_words_freq': True,
                           'word_lengths_freq': True,
                           'sentence_lengths_freq': True,
                           'pos': True,
                           'DVMA': True,
                           'DVSA': False,
                           'DVEX': False,
                           'DVL2': False,
                           'SQ': True,
                           'SQ_ngrams': [3, 7],
                           'SQ_selection_ratio': 0
                           # proportion of SQ features to be taken: 1: all, float: proportion, 0: optimized (only for 1fold)
                           }
        assert features_params[
                   'SQ_selection_ratio'] != 0 or cv_method == '1fold', 'SQ selection optimization only for 1fold method'
        SVM_classification(dataset, features_params, pickle_path, cv_method=cv_method)

    else:
        # parameters for NN experiments
        NN_params = {'DVMA': True,
                     'DVSA': True,
                     'DVEX': True,
                     'DVL2': True,
                     'FAKE': False,
                     'SQ': True
                     }
        NN_classification(dataset, NN_params, learner, dataset_name, n_sent, pickle_path, batch_size=64)
