import re
import os
import numpy as np
import pathlib
from general.helpers import splitter, metric_scansion


class dataset_KabalaCorpusA:
    def __init__(self, dir_path="../dataset/KabalaCorpusA", n_sent=10, cleaning=False):
        """
        :param dir_path: path to the dataset directory, default: "../dataset/KabalaCorpusA"
        :param n_sent: number of sentences forming a fragment, default: 10
        :param cleaning: trigger the custom cleaning of the texts, default: False
        """
        if not os.path.exists(dir_path):
            print('Dataset not found!')
            exit(0)

        # if cleaning is requested, the files are cleaned
        if cleaning:
            print('----- CLEANING TEXTS -----')
            for file in os.listdir(dir_path):
                file_path = dir_path + '/' + file
                _clean_texts(file_path)
            print('----- CLEANING COMPLETE -----')

        print('----- CREATING DATASET -----')
        self.titles = os.listdir(dir_path)
        self.authors = np.unique([title.split(' ', 1)[0] for title in self.titles]).tolist()
        self.data = []
        self.data_cltk = []
        self.authors_labels = []
        self.titles_labels = []
        for i, file in enumerate(self.titles):
            file_path = dir_path + '/' + file
            text = open(file_path, "r").read()
            author = self.authors.index(file.split(' ', 1)[0])  # get author index by splitting the file name
            fragments = splitter(text, n_sent)
            self.data.append(fragments)
            self.data_cltk.append(metric_scansion(fragments))
            # add corresponding title label, one for each fragment
            self.titles_labels.append([i] * len(fragments))
            if author is not None:
                # add corresponding author labels, one for each fragment
                self.authors_labels.append([author] * len(fragments))


# clean the text (modify the document)
def _clean_texts(file_path):
    text = open(file_path, "r", errors='ignore').read()
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub('\[.*?\]', "", text)
    text = re.sub('[0-9]', "", text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.lower()
    text = text.replace('v', 'u')
    text = text.replace('j', 'i')
    text = re.sub('\.\s+(?=\.)|\.\.+', "", text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("\(|\)|\[|\]", "", text)
    text = re.sub("\—|\–|\-|\_", "", text)
    text = re.sub("\‹|\›|\»|\«|\=|\/|\\|\~|\§|\*|\#|\@|\^|\“|\”", "", text)
    text = re.sub("\&dagger;|\&amacr;|\&emacr;|\&imacr;|\&omacr;|\&umacr;|\&lsquo;|\&rsquo;|\&rang;|\&lang;|\&lsqb;", "", text)
    text = re.sub("\?|\!|\:|\;", ".", text)
    text = text.replace("'", "")
    text = text.replace('"', '')
    text = text.replace(".,", ".")
    text = text.replace(",.", ".")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = re.sub('(\.)+', ".", text)
    text = re.sub('(\,)+', "", text)
    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    with open(file_path, "w") as f:
        f.write(text)
