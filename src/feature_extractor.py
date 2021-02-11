from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from cltk.prosody.latin.macronizer import Macronizer
from cltk.prosody.latin.scanner import Scansion
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize
import itertools
import multiprocessing
from joblib import Parallel, delayed

# ------------------------------------------------------------------------
# support functions
# ------------------------------------------------------------------------

# kestemont_2015 = ['et', 'e', 'quoniam', 'contra', 'qui', 'vel', 'in', 'aut', 'quasi', dum',
# 'pro', 'idem', 'scilicet', 'non', 'quam', 'super',
# 'velut', 'autem', 'ante', 'nunc', 'iam', 'ad', 'ne', 'semper', 'apud', 'usque', 'hic', 'ac',
# 'quantum', 'sed', 'enim', 'ut', 'etiam', 'sive', 'de', 'unde', 'inter', 'a', 'sicut', 'quidem',
# 'videlicet', 'cum', 'tam', 'magis', 'tunc', 'quod', 'ita', 'propter', 'ipse', 'tamen', 'quoque',
# 'ergo', 'atque', 'si', 'sine', 'per', 'nisi', 'post', 'sic', 'adhuc', 'quia', 'ubi', 'licet', 'nec']

# kestemont_missing_function_words = ['quoniam', 'quam', 'semper', 'licet', 'uidelicet', 'quoque', 'uelut']
# kestemont_missing_pronouns = ['qui', 'hic', 'ipse', 'quod', 'quantum']

latin_function_words = ['et', 'in', 'de', 'ad', 'non', 'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi', 'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur',
                        'circa', 'quidem', 'supra', 'ante', 'adhuc', 'seu', 'apud', 'olim', 'statim', 'satis', 'ob',
                        'quoniam', 'postea', 'nunquam', 'semper', 'licet', 'uidelicet', 'quoque', 'uelut', 'quot']


# return list of function words
def get_function_words(lang):
    if lang == 'latin':
        return latin_function_words
    elif lang in ['english', 'spanish', 'italian']:
        return stopwords.words(lang)
    else:
        raise ValueError('{} not in scope!'.format(lang))


# tokenize text
def tokenize(text):
    unmod_tokens = nltk.word_tokenize(text)
    return [token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)]

# transform text into metric scansion
# macronizing and scanning the texts
#to do: parallelize
def metric_scansion(documents):
    macronizer = Macronizer('tag_ngram_123_backoff')
    scanner = Scansion(clausula_length=10000) # clausula_length was 13, it didn't get the string before that point (it goes backward)
    scanned_texts = [scanner.scan_text(macronizer.macronize_text(doc)) for doc in documents]
    scanned_texts = [''.join(scanned_text) for scanned_text in scanned_texts]  # concatenate the sentences
    return scanned_texts


# def _get_parallel_slices(n_tasks, n_jobs=-1):
#     if n_jobs == -1:
#         n_jobs = multiprocessing.cpu_count()
#     batch = int(n_tasks / n_jobs)
#     remainder = n_tasks % n_jobs
#     return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in
#             range(n_jobs)]
#
# def _parallelize(func, args, n_jobs):
#     args = np.asarray(args)
#     slices = _get_parallel_slices(len(args), n_jobs)
#     results = Parallel(n_jobs=n_jobs)(delayed(func)(args[slice_i]) for slice_i in slices)
#     return list(itertools.chain.from_iterable(results))


# ------------------------------------------------------------------------
# feature extraction methods
# ------------------------------------------------------------------------

# extract the frequency (L1x1000) of each function word used in the documents
def _function_words_freq(documents, lang):
    features = []
    function_words = get_function_words(lang)
    for text in documents:
        mod_tokens = tokenize(text)
        freqs = nltk.FreqDist(mod_tokens)
        nwords = len(mod_tokens)
        funct_words_freq = [freqs[function_word] / nwords for function_word in function_words]
        features.append(funct_words_freq)
    f = csr_matrix(features)
    return f


# extract the frequencies (L1x1000) of the words' lengths used in the documents,
# following the idea behind Mendenhall's Characteristic Curve of Composition
def _word_lengths_freq(documents, upto=26):
    features = []
    for text in documents:
        mod_tokens = tokenize(text)
        nwords = len(mod_tokens)
        tokens_len = [len(token) for token in mod_tokens]
        tokens_count = []
        for i in range(1, upto):
            tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
        features.append(tokens_count)
    f = csr_matrix(features)
    return f


# extract lengths of the sentences, ie. number of words in the sentence
def _sentence_lengths_freq(documents, min=1, max=101):
    features = []
    for text in documents:
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        nsent = len(sentences)
        sent_len = []
        sent_count = []
        for sentence in sentences:
            mod_tokens = tokenize(sentence)
            sent_len.append(len(mod_tokens))
        for i in range(min, max):
            sent_count.append((sum(j >= i for j in sent_len)) / nsent)
        features.append(sent_count)
    f = csr_matrix(features)
    return f


# vectorize the documents with tfidf and select the best features
def _vector_select(doc_train, doc_test, y_train, type, min, max, feature_selection_ratio):
    vectorizer = TfidfVectorizer(analyzer=type, ngram_range=(min, max), sublinear_tf=True)
    f_train = vectorizer.fit_transform(doc_train)
    f_test = vectorizer.transform(doc_test)
    if feature_selection_ratio != 1:
        num_feats = int(f_train.shape[1] * feature_selection_ratio)  # number of selected features (must be int)
        selector = SelectKBest(chi2, k=num_feats)
        f_train = selector.fit_transform(f_train, y_train)
        f_test = selector.transform(f_test)
    return f_train, f_test


# ------------------------------------------------------------------------
# class for the extraction of features
# ------------------------------------------------------------------------
# class FeatureExtractor:
class FeatureExtractor:
    def __init__(self, doc_train, doc_test, y_train, y_test, feature_selection_ratio=1,
                 function_words_freq='latin',
                 word_lengths_freq=True,
                 sentence_lengths_freq=True,
                 word_ngrams=False,
                 word_ngrams_range=[2, 2],
                 char_ngrams=False,
                 char_ngrams_range=[3, 3],
                 syll_ngrams=False,
                 syll_ngrams_range=[3, 3]):

        """
        For each feature type, the corresponding function is called and a csr_matrix is created.
        The matrix is normalized through l2.
        The matrix is then added orizontally (hstack) to the final matrix.
        Train and test are kept separate to properly fit on training set for n-grams vectorization and feature selection.
        :param doc_train: documents for training
        :param doc_test: documents for test
        :param y_train: labels for training
        :param y_test: labels for test
        :param feature_selection_ratio: if not 1, the specific percentage of features is selected through chi2;
                                        only for n-grams feature types (selection done separately).
        :param function_words_freq: not selected if None, otherwise it takes the language of interest
        """

        # documents
        self.doc_train = doc_train
        self.doc_test = doc_test
        self.y_train = y_train
        self.y_test = y_test

        # final matrixes of features
        # initialize the right number of rows, or hstack won't work
        self.X_train = csr_matrix((len(doc_train), 0))
        self.X_test = csr_matrix((len(doc_test), 0))


        if function_words_freq is not None:
            f = normalize(_function_words_freq(self.doc_train, function_words_freq))
            self.X_train = hstack((self.X_train, f))
            f = normalize(_function_words_freq(self.doc_test, function_words_freq))
            self.X_test = hstack((self.X_test, f))
            print(f'task function words (#features={f.shape[1]}) [Done]')

        if word_lengths_freq:
            f = normalize(_word_lengths_freq(self.doc_train))
            self.X_train = hstack((self.X_train, f))
            f = normalize(_word_lengths_freq(self.doc_test))
            self.X_test = hstack((self.X_test, f))
            print(f'task word lengths (#features={f.shape[1]}) [Done]')

        if sentence_lengths_freq:
            f = normalize(_sentence_lengths_freq(self.doc_train))
            self.X_train = hstack((self.X_train, f))
            f = normalize(_sentence_lengths_freq(self.doc_test))
            self.X_test = hstack((self.X_test, f))
            print(f'task sentence lengths (#features={f.shape[1]}) [Done]')

        if char_ngrams:
            f_train, f_test = _vector_select(self.doc_train, self.doc_test, self.y_train, 'char',
                                             char_ngrams_range[0], char_ngrams_range[1], feature_selection_ratio)
            self.X_train = hstack((self.X_train, csr_matrix(f_train)))
            self.X_test = hstack((self.X_test, csr_matrix(f_test)))
            print(f'task character n-grams (#features={f_train.shape[1]}) [Done]')

        if syll_ngrams:
            scanned_train = metric_scansion(doc_train)
            scanned_test = metric_scansion(doc_test)
            f_train, f_test = _vector_select(scanned_train, scanned_test, self.y_train, 'char',
                                             syll_ngrams_range[0], syll_ngrams_range[1], feature_selection_ratio)
            self.X_train = hstack((self.X_train, csr_matrix(f_train)))
            self.X_test = hstack((self.X_test, csr_matrix(f_test)))
            print(f'task syllables n-grams (#features={f_train.shape[1]}) [Done]')

        print("Training shape: ", self.X_train.shape)
        print("Test shape: ", self.X_test.shape)
