import re
import urllib.request, urllib.error, urllib.parse
import os, sys
from general.helpers import splitter, metric_scansion
import pathlib


# list of authors that will be added in the dataset
authors = ['Vitruvius', 'Cicero', 'Seneca', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius', 'Servius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Florus', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus', 'Sallustius',
           'Seneca_maior', 'Sidonius_Apollinaris', 'Cornelius_Tacitus', 'Minucius_Felix',
           'Plinius_minor', 'Cornelius_Celsus', 'Beda', 'Hieronymus_Stridonensis']

# files to delete, because poetry or theatre
files_todelete = ['Cornelius_Celsus-De medicina - Ed. Daremberg.txt', 'Seneca-Agamemnon.txt', 'Seneca-Hercules Furens.txt',
                  'Seneca-Hercules Oetaeus.txt', 'Seneca-Medea.txt', 'Seneca-Octavia.txt', 'Seneca-Oedipus.txt',
                  'Seneca-Phaedra.txt', 'Seneca-Phoenissae.txt', 'Seneca-Thyestes.txt', 'Seneca-Troades.txt',
                  'Seneca_maior-Excerpta Controversiae.txt', 'Sidonius_Apollinaris-Carmina.txt']


# ------------------------------------------------------------------------
# class to create the Latinitas Antiqua dataset
# if required, the texts are downloaded and cleaned
# then, the texts are split into fragments
# the final object is made of the list of authors, the list of titles, the fragments (data), the authors labels, and the titles labels
# data, authors_labels and titles_labels are lists of lists (comes in handy later to define kfold or loo scenario)
# ------------------------------------------------------------------------

class dataset_LatinitasAntiqua:
    def __init__(self, dir_path="../dataset/LatinitasAntiqua", n_sent=10):
        """
        :param dir_path: path to the dataset directory, default: "../dataset/LatinitasAntiqua"
        :param n_sent: number of sentences forming a fragment, default: 10
        """

        # make directory 'dataset' if not existing
        os.makedirs(pathlib.Path(dir_path).parent, exist_ok=True)

        #if the directory doesn't exist, download and clean the texts
        if not os.path.exists(dir_path):
            print('----- DOWNLOADING TEXTS -----')
            _download_texts(dir_path)
            num_removed = 0
            for file in files_todelete:
                try:
                    os.remove(dir_path + '/' + file)
                    num_removed += 1
                except OSError:
                    pass
            print('Texts removed:', num_removed)
            print('----- CLEANING TEXTS -----')
            for file in os.listdir(dir_path):
                file_path = dir_path + '/' + file
                _clean_texts(file_path)
            print('----- CLEANING COMPLETE -----')

        print('----- CREATING DATASET -----')
        self.authors = authors
        self.titles = os.listdir(dir_path)
        self.data = []
        self.data_cltk = []
        self.authors_labels = []
        self.titles_labels = []
        for i, file in enumerate(self.titles):
            file_path = dir_path + '/' + file
            text = open(file_path, "r").read()
            author = authors.index(file.split('-', 1)[0])  # get author index by splitting the file name
            fragments = splitter(text, n_sent)
            self.data.append(fragments)
            self.data_cltk.append(metric_scansion(fragments))
            # add corresponding title label, one for each fragment
            self.titles_labels.append([i] * len(fragments))
            if author is not None:
                # add corresponding author labels, one for each fragment
                self.authors_labels.append([author] * len(fragments))


# download the texts from Corpus Corporum/Latinitas Antiqua
# given a list of authors, all their texts are download
def _download_texts(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print("Creation of the directory %s failed :(" % dir_path)
        sys.exit(1)
    # get list of titles for each author
    authors_titles = []
    for author in authors:
        # create link to the author's mainpage
        author_link = "http://www.mlat.uzh.ch/MLS/verzeichnis4.php?tabelle=" + author + '_cps5&id=&nummer=&lang=0&corpus=5&inframe=1'
        author_webpage = urllib.request.urlopen(author_link).read().decode('utf-8')
        # get list of titles for the single author...
        author_titles = re.findall(r'target=\"verz\">([A-Z]+.*?)</a>', author_webpage)
        # ... and add it to the list of lists authors_titles
        authors_titles.append(author_titles)
    # keep the number of downloads failed or suceeded
    num_fails = 0
    num_oks = 0
    for index, author_titles in enumerate(authors_titles):
        for title in author_titles:
            # create text file (where the download will be saved)
            text_file = open(dir_path + '/' + authors[index] + '-' + title + '.txt', 'w')
            title = title.replace(" ", "%20")  # the url doesn't work with whitespaces
            # create link to the title's mainpage
            title_link = 'http://www.mlat.uzh.ch/MLS/work_header.php?lang=0&corpus=5&table=' + \
                         authors[index] + '_cps5&title=' + title + '&id=' + authors[index] + '_cps5,' + title
            title_webpage = urllib.request.urlopen(title_link).read().decode('utf-8')
            # get the download link (xml format)
            download_link = re.search(r'download(.*?)\.xml\&xml=1', title_webpage).group(0)
            download_link = 'http://www.mlat.uzh.ch/' + download_link
            text_page = urllib.request.urlopen(download_link).read().decode('utf-8')
            # if the link is wrong, it will simply give a 'could not open' kind of webpage
            if 'could not open' in text_page:
                num_fails += 1
                print(download_link)
            else:
                num_oks += 1
                text_file.write(text_page)
            text_file.close()
    print('Texts downloaded:', num_oks)
    print('Failed attempts:', num_fails)

# clean the text (modify the document)
def _clean_texts(file_path):
    text = open(file_path, "r").read()
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub('<META(.|\n)*<\/teiHeader>', "", text)
    text = re.sub('<head(?:(?!<head|\/head)[\s\S])*\/head>', "", text)
    text = re.sub('&quot\;(?:(?!&quot\;|&quot\;)[\s\S])*&quot\;', "", text)
    text = re.sub('<quote(?:(?!<quote|\/quote)[\s\S])*\/quote>', "", text)
    text = re.sub('<app(?:(?!<app|\/app)[\s\S])*\/app>', "", text)
    text = re.sub('<bibl(?:(?!<bibl|\/bibl)[\s\S])*\/bibl>', "", text)
    text = re.sub('<foreign(?:(?!<foreign|\/foreign)[\s\S])*\/foreign>', "", text)
    text = re.sub('<note(?:(?!<note|\/note)[\s\S])*\/note>', "", text)
    text = re.sub('<date(?:(?!<date|\/date)[\s\S])*\/date>', "", text)
    text = re.sub('<ref(?:(?!<rf|\/rf)[\s\S])*\/ref>', "", text)
    text = re.sub('<i(?:(?!<i|\/i)[\s\S])*\/i>', "", text)
    text = re.sub('<argument(?:(?!<argument|\/argument)[\s\S])*\/argument>', "", text)
    text = re.sub('<[^<]*?>', "", text)
    text = re.sub('[0-9]', "", text)
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
