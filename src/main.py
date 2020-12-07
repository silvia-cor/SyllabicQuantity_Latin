import dataset_loader, feature_extractor
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

#sometimes the learning method does not converge; this is to suppress a lot of warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

#list of authors that will be added in the dataset
authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Petronius', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Plinius_minor', 'Beda']

dataset_path = "../dataset"  # change here for directory location

#just change values for download and cleaning
dataset = dataset_loader.DatasetBuilder(authors, dataset_path,
                                        download=False, cleaning=False, n_sentences=7)

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
i=0
y_preds = [] #series of predictions
y_tests = [] #series of true labels

#k-fold cross validation
for train_index, test_index in k_fold.split(dataset.data, dataset.authors_labels):
    i += 1
    x_train = [dataset.data[index] for index in train_index]
    x_test = [dataset.data[index] for index in test_index]
    y_train = [dataset.authors_labels[index] for index in train_index]
    y_test = [dataset.authors_labels[index] for index in test_index]

    print(f'----- K-FOLD EXPERIMENT {i} -----')
    print('FEATURE EXTRACTION')
    features = feature_extractor.FeatureExtractor(x_train, x_test, y_train, y_test, syll_ngrams=False)
    X_train, X_test = features.X_train, features.X_test

    print('CLASSIFICATION')
    #param_grid= {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    param_grid= {'C': [1]}
    #cls = SVC(kernel='linear', random_state=42)
    cls = GridSearchCV(LogisticRegression(max_iter=100), param_grid)
    cls.fit(X_train, y_train)
    print('Best C:', cls.best_params_['C'])
    y_pred = cls.predict(X_test)
    f1 = f1_score(y_test,y_pred, average='weighted')
    print(f'F1: {f1:.3f}')
    y_preds.extend(y_pred)
    y_tests.extend(y_test)


print('----- FINAL SCORE -----')
macro_f1 = f1_score(y_tests, y_preds, average='macro')
micro_f1 = f1_score(y_tests, y_preds, average='micro')
print(f'Macro-F1: {macro_f1:.3f}')
print(f'Micro-F1: {micro_f1:.3f}')



