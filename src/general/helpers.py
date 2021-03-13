import nltk
from nltk.corpus import stopwords
from cltk.prosody.latin.macronizer import Macronizer
from cltk.prosody.latin.scanner import Scansion
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import itertools

# ------------------------------------------------------------------------
# functions for managing the text processing
# ------------------------------------------------------------------------

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
    return [token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)] # checks whether all the chars are alphabetic


# ------------------------------------------------------------------------
# functions to distort the text
# ------------------------------------------------------------------------

# # transform texts into fake metric scansion
# # requires entire dataset
# def fake_metric(docs):
#     anal_words = CountVectorizer(analyzer='word', token_pattern=r'[^\s]+', ngram_range=(1, 1)).fit(docs)
#     vocabulary = anal_words.vocabulary_
#     for vocab in vocabulary:
#         n = len(vocab) // 3 if len(vocab) // 3 > 1 else 1
#         anal_words.vocabulary_[vocab] = np.random.choice(np.asarray(['u', 'x', '-']), size=n, replace=True).tolist()
#     dis_texts = []
#     for doc in docs:
#         mod_tokens = tokenize(doc)
#         tokenizer = anal_words.build_analyzer()
#         for doc in docs:
#             dis_text = [vocabulary.get(item, vocabulary.get('<unk>')) for item in tokenizer(doc)]
#             encoded_texts.append(encoded_text)
#         dis_tokens = [token if token in function_words else ('*' * len(token)) for token in mod_tokens]
#             dis_texts.append(' '.join(dis_tokens))
#         return dis_texts



# transform text into metric scansion
# macronizing and scanning the texts
#to do: parallelize
def metric_scansion(docs):
    macronizer = Macronizer('tag_ngram_123_backoff')
    scanner = Scansion(clausula_length=100000) # clausula_length was 13, it didn't get the string before that point (it goes backward)
    scanned_texts = [scanner.scan_text(macronizer.macronize_text(doc)) for doc in docs]
    scanned_texts = [''.join(scanned_text) for scanned_text in scanned_texts]  # concatenate the sentences
    return scanned_texts


# DV-MA text distortion method from Stamatatos_2018:
# Every word not in function_words is masked by replacing each of its characters with an asterisk (*).
# for character embedding
def dis_DVMA(docs, function_words):
    dis_texts = []
    for doc in docs:
        mod_tokens = tokenize(doc)
        dis_tokens = [token if token in function_words else ('*' * len(token)) for token in mod_tokens]
        dis_texts.append(' '.join(dis_tokens))
    return dis_texts

# DV-SA text distortion method from Stamatatos_2018:
# Every word not in function_words is replaced with an asterisk (*).
# for word embedding
def dis_DVSA(docs, function_words):
    dis_texts = []
    for doc in docs:
        mod_tokens = tokenize(doc)
        dis_tokens = [token if token in function_words else '*' for token in mod_tokens]
        dis_texts.append(' '.join(dis_tokens))
    return dis_texts

# DV-EX text distortion method from Stamatatos_2018:
# Every word not in function_words is masked by replacing each of its characters with an asterisk (*), except first and last one.
# Words of len 2 or 1 remain the same.
# for character embedding
def dis_DVEX(docs, function_words):
    dis_texts = []
    for doc in docs:
        mod_tokens = tokenize(doc)
        dis_tokens = [token if token in function_words or len(token) == 1
                      else token[0] + ('*' * (len(token) - 2)) + token[len(token) - 1] for token in mod_tokens]
        dis_texts.append(' '.join(dis_tokens))
    return dis_texts

# DV-EX text distortion method from Stamatatos_2018:
# Every word not in function_words is masked by replacing each of its characters with an asterisk (*), except last 2 ones.
# Words of len 2 or 1 remain the same.
# for character embedding
def dis_DVL2(docs, function_words):
    dis_texts = []
    for doc in docs:
        mod_tokens = tokenize(doc)
        dis_tokens = [token if token in function_words or len(token) == 1
                      else ('*' * (len(token) - 2)) + token[len(token) - 2]  + token[len(token) - 1] for token in mod_tokens]
        dis_texts.append(' '.join(dis_tokens))
    return dis_texts


# ------------------------------------------------------------------------
# functions to split the text into fragments of (n_sentences) sentences
# ------------------------------------------------------------------------

#split text in fragments made of (n_sentences) sentences
#also adds the entire text in the first index
def splitter(text, n_sentences):
    text_fragments = []
    text_fragments.append(text) #add whole text
    sentences = _split_sentences(text)
    text_fragments.extend(_group_sentences(sentences, n_sentences))
    return text_fragments

# split text into single sentences
def _split_sentences(text):
    # strip() removes blank spaces before and after string
    sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
    for i, sentence in enumerate(sentences):
        mod_tokens = tokenize(sentence)
        if len(mod_tokens) < 5:  # if the sentence is less than 5 words long, it is...
            if i < len(sentences) - 1:
                sentences[i + 1] = sentences[i] + ' ' + sentences[i + 1]  # combined with the next sentence
            else:
                sentences[i - 1] = sentences[i - 1] + ' ' + sentences[i]  # or the previous one if it was the last sentence
            sentences.pop(i)  # and deleted as a standalone sentence
    return sentences

# group sentences into fragments of window_size sentences
# not overlapping
def _group_sentences(sentences, window_size):
    new_fragments = []
    nbatches = len(sentences) // window_size
    if len(sentences) % window_size > 0:
        nbatches += 1
    for i in range(nbatches):
        offset = i * window_size
        new_fragments.append(' '.join(sentences[offset:offset + window_size]))
    return new_fragments

# ------------------------------------------------------------------------
# functions to prepare the dataset for k-fold or loo cross-validation
# ------------------------------------------------------------------------

# prepares dataset for k-fold-cross-validation
# takes out first element in data and labels (which is the whole text) and transform in numpy array
def data_for_kfold(dataset):
    authors = np.array(dataset.authors)
    titles = np.array(dataset.titles)
    data = np.array([sub_list[i] for sub_list in dataset.data for i in range(1, len(sub_list))])
    authors_labels = np.array([sub_list[i] for sub_list in dataset.authors_labels for i in range(1, len(sub_list))])
    titles_labels = np.array([sub_list[i] for sub_list in dataset.titles_labels for i in range(1, len(sub_list))])
    return authors, titles, data, authors_labels, titles_labels

# prepares dataset for loo (transform everything into numpy array)
def data_for_loo(dataset):
    authors = np.array(dataset.authors)
    titles = np.array(dataset.titles)
    data = np.array(list(itertools.chain.from_iterable(dataset.data)), dtype=object)
    authors_labels = np.array(list(itertools.chain.from_iterable(dataset.authors_labels)))
    titles_labels = np.array(list(itertools.chain.from_iterable(dataset.titles_labels)))
    return authors, titles, data, authors_labels, titles_labels


