from dataset_loader import DatasetBuilder
import significance
from utils import pickled_resource
from feature_extractor import metric_scansion


# list of authors that will be added in the dataset
authors = ['Vitruvius', 'Cicero', 'Seneca', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Florus', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus', 'Sallustius',
           'Seneca_maior', 'Sidonius_Apollinaris', 'Cornelius_Tacitus', 'Minucius_Felix',
           'Plinius_minor', 'Cornelius_Celsus', 'Beda', 'Hieronymus_Stridonensis']

dataset_path = "../dataset"  # change here for directory location


dataset = pickled_resource("../pickles/dataset_8sent.pickle",
                 DatasetBuilder, authors, dataset_path, download=False, cleaning=False, n_sentences=8)

features_params = {'feature_selection_ratio': 1,
                                          'function_words_freq': 'latin',
                                          'word_lengths_freq': True,
                                          'sentence_lengths_freq': True,
                                          'word_ngrams': False,
                                          'word_ngrams_range': [2, 2],
                                          'char_ngrams': False,
                                          'char_ngrams_range': [3, 3],
                                          'syll_ngrams': False,
                                          'syll_ngrams_range': [3, 3]}

file_path = '../pickles/kfold_experiments.pickle'
# file_path = '../pickles/loo_experiments.pickle'
significance.exp_create_base(dataset, file_path, features_params, 'kfold')
#
# feature_sels = [0.1, 0.3]
# mins = [8, 9, 10]
# #maxs = [7]
#
# for feature_sel in feature_sels:
#     for min in mins:
#         #for max in maxs:
#             #if (min > max):
#                 #pass
#             #else:
#                 features_params = {'feature_selection_ratio': feature_sel,
#                                    'function_words_freq': 'latin',
#                                    'word_lengths_freq': True,
#                                    'sentence_lengths_freq': True,
#                                    'word_ngrams': False,
#                                    'word_ngrams_range': [2, 2],
#                                    'char_ngrams': False,
#                                    'char_ngrams_range': [3, 3],
#                                    'syll_ngrams': True,
#                                    'syll_ngrams_range': [min, min]}
#
#                 significance.exp_significance_test(dataset, file_path, features_params, 'loo')
#


