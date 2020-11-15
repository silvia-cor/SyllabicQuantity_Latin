import dataset_loader

#list of authors that will be added in the dataset2
authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Petronius', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Plinius_minor', 'Beda']

dataset_path = "../dataset"  # change here for directory location

#just change values for download and cleaning
dataset = dataset_loader.DatasetBuilder(authors, dataset_path,
                                        download=False, cleaning=False, n_sentences=5)

X, y = dataset.data, dataset.authors_labels