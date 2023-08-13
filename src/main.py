from general.utils import pickled_resource
from SVM_classification import SVM_classification
from NN_classification import NN_classification
import importlib
import argparse
from distutils.util import strtobool


def run(dataset_name: str, learner: str, DVMA: bool, DVSA: bool, DVEX: bool, DVL2: bool, SQ: bool, FAKE,
        SQ_ngrams, SQ_selection_ratio, n_sent, cv_method, batch_size, datasets_path, random_state):
    """
    :param dataset_name: name of the dataset to process;
    :param learner: name of the learner to train (SVM or NN method);
    :param DVMA: whether to add the 'Distorted View: Multiple Asterisks' features/channel;
    :param DVSA: whether to add the 'Distorted View: Single Asterisk' features/channel;
    :param DVEX: whether to add the 'Distorted View: Exterior Characters' features/channel;
    :param DVL2: whether to add the 'Distorted View: Last 2' features/channel;
    :param SQ: whether to add the SQ features/channel;
    :param FAKE: (only for NN methods) whether to add the FAKE channel;
    :param SQ_ngrams: (only for SQ) range of ngrams to compute for SQ features [min, max];
    :param SQ_selection_ratio: (only for SQ) selection ratio for SQ features (1: no selection |
                                float: proportion of features to select | 0: selection optiomized via 1fold CV);
    :param n_sent: the documents in the dataset divided into n sentences;
    :param cv_method: (only for SVM) method for crossvalidation;
    :param batch_size: (only for NN methods) the batch size;
    :param datasets_path: the path to the main directory for the datasets files;
    :param random_state: the seed for the random processes (for reproducibility).
    """
    assert learner in ['svm', 'cnn_shallow_ensemble'], \
        'Learner not implemented. Options: svm, cnn_shallow_ensemble.'
    assert cv_method in ['1fold', 'kfold', 'loo'], 'CV method must be either 1fold, kfold or loo.'
    assert dataset_name in ['LatinitasAntiqua', 'KabalaCorpusA', 'MedLatin'], \
        'Dataset unknown, the dataset available are: LatinitasAntiqua, KabalaCorpusA, MedLatin.'

    # paths to store the pickles
    data_path = f"../pickles/{dataset_name}/{dataset_name}_{n_sent}sent.pickle"
    pickle_path = f'../pickles/{dataset_name}/{learner}_{cv_method}_{n_sent}sent.pickle'

    # the dataset is loaded based on the name given
    module = importlib.import_module(f'dataset_prep.{dataset_name}_prep')
    dataset = pickled_resource(data_path, getattr(module, f'dataset_{dataset_name}'),
                               dir_path=datasets_path + dataset_name, n_sent=n_sent)

    # features to be used (BaseFeatures are always added)
    assert SQ_selection_ratio != 0 or cv_method == '1fold', 'SQ selection optimization only for 1fold method'

    params = {'DVMA': DVMA,
              'DVSA': DVSA,
              'DVEX': DVEX,
              'DVL2': DVL2,
              'FAKE': FAKE,
              'SQ': SQ,
              'SQ_ngrams': SQ_ngrams,
              'SQ_selection_ratio': SQ_selection_ratio
              }

    if learner == 'svm':
        # SVM experiment
        SVM_classification(dataset, params, pickle_path, cv_method=cv_method, random_state=random_state)
    else:
        # NN experiment
        NN_classification(dataset, params, learner, dataset_name, n_sent, pickle_path,
                          batch_size=batch_size, random_state=random_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset_name', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('-learner', type=str, required=True, help='name of the learner to train')
    parser.add_argument('-DVMA', type=lambda x: bool(strtobool(x)), required=True,
                        help='whether to add the DVMA features/channel')
    parser.add_argument('-DVSA', type=lambda x: bool(strtobool(x)), required=True,
                        help='whether to add the DVSA features/channel')
    parser.add_argument('-DVEX', type=lambda x: bool(strtobool(x)), required=True,
                        help='whether to add the DVEX features/channel')
    parser.add_argument('-DVL2', type=lambda x: bool(strtobool(x)), required=True,
                        help='whether to add the DVL2 features/channel')
    parser.add_argument('-SQ', type=lambda x: bool(strtobool(x)), required=True,
                        help='whether to add the SQ features/channel')
    parser.add_argument('--FAKE', type=lambda x: bool(strtobool(x)), required=False, default=False,
                        help='whether to add the FAKE features/channel')
    parser.add_argument('--SQ_ngrams_min', type=int, required=False, default=3,
                        help='minimum value for SQ ngrams range')
    parser.add_argument('--SQ_ngrams_max', type=int, required=False, default=7,
                        help='maximum value for SQ ngrams range')
    parser.add_argument('--SQ_selection_ratio', required=False, default=0, help='selection ratio for SQ features '
                                                                                '(1: no selection | '
                                                                                'float: proportion of features to select |'
                                                                                ' 0: selection optiomized via 1fold CV)')
    parser.add_argument('--n_sent', type=int, required=False, default=10, help='number of sentences to divide the '
                                                                               'documents into')
    parser.add_argument('--cv_method', type=str, required=False, default='1fold', help='SVM method for crossvalidation')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size for NN experiments')
    parser.add_argument('--datasets_path', type=str, required=False, default='../dataset/',
                        help='path to the main directory for the files of the datasets; each dataset should be put in '
                             'a separate sub-directory with the same name as the dataset')
    parser.add_argument('--random_state', type=int, required=False, default=42, help='seed for random processes')
    args = parser.parse_args()

    print(f'----- CONFIGURATION -----')
    d = vars(args)
    for k in d:
        print(k, ':', d[k])

    run(args.dataset_name, args.learner, DVMA=args.DVMA, DVSA=args.DVSA, DVEX=args.DVEX, DVL2=args.DVL2,
        SQ=args.SQ, FAKE=args.FAKE, SQ_ngrams=[args.SQ_ngrams_min, args.SQ_ngrams_max],
        SQ_selection_ratio=args.SQ_selection_ratio, n_sent=args.n_sent, cv_method=args.cv_method,
        batch_size=args.batch_size, datasets_path=args.datasets_path, random_state=args.random_state)
