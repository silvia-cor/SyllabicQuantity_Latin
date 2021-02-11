import re
import nltk
import urllib.request, urllib.error, urllib.parse
import os, sys
import numpy as np
import itertools


# ------------------------------------------------------------------------
# class to download, clean and split the texts, building the dataset
# ------------------------------------------------------------------------
class DatasetBuilder:
    def __init__(self, authors, dir_path="../dataset", download=False, cleaning=False, n_sentences=10):

        """
        :param authors: list of authors names
        :param dir_path: path to the dataset directory, default: "../dataset"
        :param download: whether it is necessary to download the texts, default: False/No
        :param download: whether it is necessary to clean the texts, default: False/No
        :param n_sentences: number of sentences that will make a document sample, default: 5
        """

        self.authors = authors
        self.dir_path = dir_path
        self.n_sentences = n_sentences
        self.data = []  # text samples
        self.titles_list = [] # titles of the works
        self.authors_labels = []  # labels with authors indexes
        self.titles_labels = []  # labels with titles indexes

        if download:  # if download == True, download the texts
            self._download_texts()

        if cleaning:  # if cleaning == True, clean the texts
            print('----- CLEANING TEXTS -----')
            for file in os.listdir(self.dir_path):
                file_path = self.dir_path + '/' + file
                self._remove_tags(file_path)
            print('----- CLEANING COMPLETE -----')

        # creates the fragments and the corresponding labels
        print('----- CREATING DATASET -----')
        for i, file in enumerate(os.listdir(self.dir_path)):
            file_path = self.dir_path + '/' + file
            text = open(file_path, "r").read()
            author = authors.index(file.split('-',1)[0]) # get author index by splitting the file name
            self.titles_list.append(file) #append title
            self._splitter(text, author, i)


    # STEP 1: download the texts from Corpus Corporum
    def _download_texts(self):
        print('----- DOWNLOADING TEXTS -----')
        try:
            os.mkdir(self.dir_path)
        except OSError:
            print("Creation of the directory %s failed :(" % self.dir_path)
            sys.exit(1)
        # get list of titles for each author
        authors_titles = []
        for author in self.authors:
            # create link to the author's mainpage
            author_link = "http://www.mlat.uzh.ch/MLS/verzeichnis4.php?tabelle=" + \
                          author + '_cps5&id=&nummer=&lang=0&corpus=5&inframe=1'
            author_webpage = urllib.request.urlopen(author_link).read().decode('utf-8')
            # get list of titles for the single author...
            author_titles = re.findall(r'target=\"verz\">([A-Z]+.*?)</a>', author_webpage)
            # ... and add it to the list of lists authors_titles
            authors_titles.append(author_titles)
        # keep the number of downloads failed or suceeded
        n_fails = 0
        n_oks = 0
        for index, author_titles in enumerate(authors_titles):
            for title in author_titles:
                # create text file (where the download will be saved)
                text_file = open(self.dir_path + '/' + self.authors[index] + '-' + title + '.txt', 'w')
                title = title.replace(" ", "%20")  # the url doesn't work with whitespaces
                # create link to the title's mainpage
                title_link = 'http://www.mlat.uzh.ch/MLS/work_header.php?lang=0&corpus=5&table=' + \
                             self.authors[index] + '_cps5&title=' + title + '&id=' + self.authors[
                                 index] + '_cps5,' + title
                title_webpage = urllib.request.urlopen(title_link).read().decode('utf-8')
                # get the download link (xml format)
                download_link = re.search(r'download(.*?)\.xml\&xml=1', title_webpage).group(0)
                download_link = 'http://www.mlat.uzh.ch/' + download_link
                text_page = urllib.request.urlopen(download_link).read().decode('utf-8')
                # if the link is wrong, it will simply give a 'could not open' kind of webpage
                if 'could not open' in text_page:
                    n_fails += 1
                    print(download_link)
                else:
                    n_oks += 1
                    text_file.write(text_page)
                text_file.close()
        print('Texts downloaded:', n_oks)
        print('Failed attempts:', n_fails)

    # STEP 2: clean the text (modify the document)
    def _remove_tags(self, path_file):
        text = open(path_file, "r").read()
        text = re.sub("\n+", " ", text)
        text = re.sub("\s+", " ", text)
        text = re.sub('<META(.|\n)*<\/teiHeader>', "", text)
        text = re.sub('<head(?:(?!<head|\/head)[\s\S])*\/head>', "", text)
        text = re.sub('&quot\;(?:(?!&quot\;|&quot\;)[\s\S])*&quot\;',"",text)
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
        text = re.sub('\.\s+(?=\.)|\.\.+',"",text)
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
        text = re.sub('(\,)+', ",", text)
        text = text.replace("á", "a")
        text = text.replace("é", "e")
        text = text.replace("í", "i")
        text = text.replace("ó", "o")
        text = re.sub("\n+", " ", text)
        text = re.sub("\s+", " ", text)
        with open(path_file, "w") as f:
            f.write(text)

    # STEP 3: divide the text into sentences
    def _splitter(self, text, author, title):
        text_fragments = []
        text_fragments.append(text) #add whole text
        sentences = self.__split_sentences(text)
        text_fragments.extend(self.__group_sentences(sentences, self.n_sentences))
        self.data.append(text_fragments)
        # add corresponding title label, one for each fragment
        self.titles_labels.append([title] * len(text_fragments))
        if author is not None:
            # add corresponding author labels, one for each fragment
            self.authors_labels.append([author] * len(text_fragments))

    # ------------------------------------------------------------------------
    # aiding functions (split and group sentences)
    # ------------------------------------------------------------------------

    # split text into sentences
    def __split_sentences(self, text):
        # strip() removes blank spaces before and after string
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        for i, sentence in enumerate(sentences):
            unmod_tokens = nltk.tokenize.word_tokenize(sentence)  # tokenizes the single sentence
            mod_tokens = ([token for token in unmod_tokens
                           if any(char.isalpha() for char in token)])  # checks whether all the chars are alphabetic
            if len(mod_tokens) < 8:  # if the sentence is less than 8 words long, it is...
                if i < len(sentences) - 1:
                    sentences[i + 1] = sentences[i] + ' ' + sentences[i + 1]  # combined with the next sentence
                else:
                    sentences[i - 1] = sentences[i - 1] + ' ' + sentences[
                        i]  # or the previous one if it was the last sentence
                sentences.pop(i)  # and deleted as a standalone sentence
        return sentences

    # group sentences into fragments of window_size sentences
    # not overlapping
    def __group_sentences(self, sentences, window_size):
        new_fragments = []
        nbatches = len(sentences) // window_size
        if len(sentences) % window_size > 0:
            nbatches += 1
        for i in range(nbatches):
            offset = i * window_size
            new_fragments.append(' '.join(sentences[offset:offset + window_size]))
        return new_fragments

    # ------------------------------------------------------------------------
    # functions to prepare for k-fold or loo cross-validation (called in classification)
    # ------------------------------------------------------------------------

    # prepares dataset for k-fold-cross-validation
    # takes out first element in data and labels (which is the whole text) and transform in numpy array
    def data_for_kfold(self):
        authors = np.array(self.authors)
        titles_list = np.array(self.titles_list)
        data = np.array([sub_list[i] for sub_list in self.data for i in range(1, len(sub_list))])
        authors_labels = np.array([sub_list[i] for sub_list in self.authors_labels for i in range(1, len(sub_list))])
        titles_labels = np.array([sub_list[i] for sub_list in self.titles_labels for i in range(1, len(sub_list))])
        return authors, titles_list, data, authors_labels, titles_labels

    # prepares dataset for loo (transform everything into numpy array)
    def data_for_loo(self):
        authors = np.array(self.authors)
        titles_list = np.array(self.titles_list)
        data = np.array(list(itertools.chain.from_iterable(self.data)), dtype=object)
        authors_labels = np.array(list(itertools.chain.from_iterable(self.authors_labels)))
        titles_labels = np.array(list(itertools.chain.from_iterable(self.titles_labels)))
        return authors, titles_list, data, authors_labels, titles_labels



