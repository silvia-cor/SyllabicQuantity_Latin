import dataset_loader, feature_extractor
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

#list of authors that will be added in the dataset2
authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Petronius', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Plinius_minor', 'Beda']

dataset_path = "../dataset"  # change here for directory location

#just change values for download and cleaning
dataset = dataset_loader.DatasetBuilder(authors, dataset_path,
                                        download=False, cleaning=False, n_sentences=5)

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
i=0
for train_index, test_index in k_fold.split(dataset.data, dataset.authors_labels):
    i += 1
    x_train = [dataset.data[index] for index in train_index]
    x_test = [dataset.data[index] for index in test_index]
    y_train = [dataset.authors_labels[index] for index in train_index]
    y_test = [dataset.authors_labels[index] for index in test_index]

    print(f'----- K-FOLD EXPERIMENT {i} -----')
    X_train, X_test = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test,
                                                feature_selection_ratio=.5,
                                              char_ngrams=False, syll_ngrams=False)
    print('Training shape: ', X_train.shape)
    print('Test shape: ', X_test.shape)

    print('----- CLASSIFICATION -----')
    cls = SVC(kernel='linear', random_state=42)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print('F1: ', f1_score(y_test, y_pred))



