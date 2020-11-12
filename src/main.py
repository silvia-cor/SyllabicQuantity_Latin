import download_dataset
import preprocessing
import os

#list of authors that will be added in the dataset
authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Petronius', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Plinius_minor', 'Beda']

dataset_path = "../dataset"  # change here for directory location

download_dataset.download(authors, dataset_path)

for file in os.listdir(dataset_path):
	preprocessing.removeTags(file)
