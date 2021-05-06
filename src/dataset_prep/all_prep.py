from dataset_prep.KabalaCorpusA_prep import dataset_KabalaCorpusA
from dataset_prep.LatinitasAntiqua_prep import dataset_LatinitasAntiqua
from dataset_prep.MedLatin_prep import dataset_MedLatin

class dataset_all:
    def __init__(self, n_sent=10):
        corpus_KabalaCorpusA = dataset_KabalaCorpusA(n_sent=n_sent)
        corpus_LatinitasAntiqua = dataset_LatinitasAntiqua(n_sent=n_sent)
        corpus_MedLatin = dataset_MedLatin(n_sent=n_sent)

        self.titles = corpus_KabalaCorpusA.titles + corpus_LatinitasAntiqua.titles + corpus_MedLatin.titles
        self.authors = corpus_KabalaCorpusA.authors + corpus_LatinitasAntiqua.authors + corpus_MedLatin.authors
        self.data = corpus_KabalaCorpusA.data + corpus_LatinitasAntiqua.data + corpus_MedLatin.data
        self.data_cltk = corpus_KabalaCorpusA.data_cltk + corpus_LatinitasAntiqua.data_cltk + corpus_MedLatin.data_cltk
        self.authors_labels = []
        self.titles_labels = []

        for corpus in [corpus_KabalaCorpusA, corpus_LatinitasAntiqua, corpus_MedLatin]:
            self.do_authors_labels(corpus)
            self.do_titles_labels(corpus)

    def do_authors_labels(self, dataset):
        for sub_list in dataset.authors_labels:
            assert all(x == sub_list[0] for x in sub_list)
            author_original = dataset.authors[sub_list[0]]
            new_label = self.authors.index(author_original)
            self.authors_labels.append([new_label] * len(sub_list))

    def do_titles_labels(self, dataset):
        for sub_list in dataset.titles_labels:
            assert all(x == sub_list[0] for x in sub_list)
            title_original = dataset.titles[sub_list[0]]
            new_label = self.titles.index(title_original)
            self.titles_labels.append([new_label] * len(sub_list))

