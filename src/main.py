import dataset_loader, classification

#list of authors that will be added in the dataset
authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Petronius', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Plinius_minor', 'Beda']

dataset_path = "../dataset"  # change here for directory location

#just change values for download and cleaning
dataset = dataset_loader.DatasetBuilder(authors, dataset_path,
                                        download=False, cleaning=False, n_sentences=10)

features_params = {'feature_selection_ratio': 1,
                   'function_words_freq': 'latin',
                   'words_lengths_freq': True,
                   'sentence_lengths_freq': True,
                   'word_ngrams': False,
                   'word_ngrams_range': [2, 2],
                   'char_ngrams': False,
                   'char_ngrams_range': [3, 3],
                   'syll_ngrams': False,
                   'syll_ngrams_range': [3, 3]}

#classification.kfold_crossval(dataset, features_params, n_splits=5)
classification.loo_crossval(dataset, features_params)





