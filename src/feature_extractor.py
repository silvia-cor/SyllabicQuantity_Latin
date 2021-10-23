from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import nltk
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize
from general.helpers import get_function_words, tokenize_nopunct, metric_scansion, dis_DVMA, dis_DVSA, dis_DVEX, \
    dis_DVL2


# ------------------------------------------------------------------------
# feature extraction methods
# ------------------------------------------------------------------------

# extract the frequency (L1x1000) of each function word used in the documents
def _function_words_freq(documents, function_words):
    features = []
    for text in documents:
        mod_tokens = tokenize_nopunct(text)
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
        mod_tokens = tokenize_nopunct(text)
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
            mod_tokens = tokenize_nopunct(sentence)
            sent_len.append(len(mod_tokens))
        for i in range(min, max):
            sent_count.append((sum(j >= i for j in sent_len)) / nsent)
        features.append(sent_count)
    f = csr_matrix(features)
    return f


# vectorize the documents with tfidf and select the best features
def _vector_select(doc_train, doc_test, y_train, min, max, SQ_selection_ratio):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(min, max), sublinear_tf=True)
    f_train = vectorizer.fit_transform(doc_train)
    f_test = vectorizer.transform(doc_test)
    if SQ_selection_ratio != 1:
        num_feats = int(f_train.shape[1] * SQ_selection_ratio)  # number of selected features (must be int)
        selector = SelectKBest(chi2, k=num_feats)
        f_train = selector.fit_transform(f_train, y_train)
        f_test = selector.transform(f_test)
    return f_train, f_test


# vectorize the documents with tfidf (only for distortion methods)
# parameters following Stamatatos_2018
def _vector_dis(doc_train, doc_test):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=5, sublinear_tf=True)
    f_train = vectorizer.fit_transform(doc_train)
    f_test = vectorizer.transform(doc_test)
    return f_train, f_test


# ------------------------------------------------------------------------
# Feature Extractor
# ------------------------------------------------------------------------
def featuresExtractor(doc_train, doc_test, cltk_train, cltk_test, y_tr, lang='latin',
                      SQ_selection_ratio=1,
                      function_words_freq=True,
                      word_lengths_freq=True,
                      sentence_lengths_freq=True,
                      DVMA=False,
                      DVSA=False,
                      DVEX=False,
                      DVL2=False,
                      SQ=False,
                      SQ_ngrams=[3, 3]):
    """
        For each feature type, the corresponding function is called and a csr_matrix is created.
        The matrix is normalized through l2.
        The matrix is then added orizontally (hstack) to the final matrix.
        Train and test are kept separate to properly fit on training set for n-grams vectorization and feature selection.
        :param doc_train: documents for training
        :param doc_test: documents for test
        :param cltk_train: metric scansion for training
        :param cltk_test: metric scansion for test
        :param y_tr: labels for training
        :param lang: language to retrieve function words, default: latin
        :param feature_selection_ratio: if not 1, the specific percentage of features is selected through chi2;
                                    only for n-grams feature types (selection done separately).
        :param function_words_freq: not selected if None, otherwise it takes the language of interest
        :param DVMA/DVSA/DVEX/DVL2: not selected if None, otherwise it takes the language of interest (for function words)
        :param SQ: not selected if None, otherwise it takes the range (for n-grams)
        """

    # final matrixes of features
    # initialize the right number of rows, or hstack won't work
    X_tr = csr_matrix((len(doc_train), 0))
    X_te = csr_matrix((len(doc_test), 0))

    fw = get_function_words(lang)

    if function_words_freq:
        f = normalize(_function_words_freq(doc_train, fw))
        X_tr = hstack((X_tr, f))
        f = normalize(_function_words_freq(doc_test, fw))
        X_te = hstack((X_te, f))
        print(f'task function words (#features={f.shape[1]}) [Done]')

    if word_lengths_freq:
        f = normalize(_word_lengths_freq(doc_train))
        X_tr = hstack((X_tr, f))
        f = normalize(_word_lengths_freq(doc_test))
        X_te = hstack((X_te, f))
        print(f'task word lengths (#features={f.shape[1]}) [Done]')

    if sentence_lengths_freq:
        f = normalize(_sentence_lengths_freq(doc_train))
        X_tr = hstack((X_tr, f))
        f = normalize(_sentence_lengths_freq(doc_test))
        X_te = hstack((X_te, f))
        print(f'task sentence lengths (#features={f.shape[1]}) [Done]')

    if DVMA:
        dis_train = dis_DVMA(doc_train, fw)
        dis_test = dis_DVMA(doc_test, fw)
        f_train, f_test = _vector_dis(dis_train, dis_test)
        X_tr = hstack((X_tr, csr_matrix(f_train)))
        X_te = hstack((X_te, csr_matrix(f_test)))
        print(f'task DVMA 3-grams (#features={f_train.shape[1]}) [Done]')

    if DVSA:
        dis_train = dis_DVSA(doc_train, fw)
        dis_test = dis_DVSA(doc_test, fw)
        f_train, f_test = _vector_dis(dis_train, dis_test)
        X_tr = hstack((X_tr, csr_matrix(f_train)))
        X_te = hstack((X_te, csr_matrix(f_test)))
        print(f'task DVSA 3-grams (#features={f_train.shape[1]}) [Done]')

    if DVEX:
        dis_train = dis_DVEX(doc_train, fw)
        dis_test = dis_DVEX(doc_test, fw)
        f_train, f_test = _vector_dis(dis_train, dis_test)
        X_tr = hstack((X_tr, csr_matrix(f_train)))
        X_te = hstack((X_te, csr_matrix(f_test)))
        print(f'task DVEX 3-grams (#features={f_train.shape[1]}) [Done]')

    if DVL2:
        dis_train = dis_DVL2(doc_train, fw)
        dis_test = dis_DVL2(doc_test, fw)
        f_train, f_test = _vector_dis(dis_train, dis_test)
        X_tr = hstack((X_tr, csr_matrix(f_train)))
        X_te = hstack((X_te, csr_matrix(f_test)))
        print(f'task DVL2 3-grams (#features={f_train.shape[1]}) [Done]')

    if SQ:
        f_train, f_test = _vector_select(cltk_train, cltk_test, y_tr, SQ_ngrams[0], SQ_ngrams[1], SQ_selection_ratio)
        X_tr = hstack((X_tr, csr_matrix(f_train)))
        X_te = hstack((X_te, csr_matrix(f_test)))
        print(f'task SQ n-grams (#features={f_train.shape[1]}) [Done]')

    return X_tr, X_te
