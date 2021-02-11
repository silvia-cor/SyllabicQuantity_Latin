import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset_loader import DatasetBuilder
from feature_extractor import tokenize, get_function_words, metric_scansion

# ------------------------------------------------------------------------
# classes for managing the dataset
# ------------------------------------------------------------------------

# pytorch Dataset class (for DataLoader)
class NN_BaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# main class: create and divides (train/test, batches) the dataset
class NN_DatasetBuilder:
    def __init__(self, authors, dataset_path="../dataset", download=False, cleaning=False, n_sentences=10, batch_size =128):
        # get the data (no whole documents)
        dataset = DatasetBuilder(authors, dataset_path, download=download, cleaning=cleaning, n_sentences=n_sentences)
        self.function_words = get_function_words('latin') #for distortion techniques
        self.authors, self.titles_list, self.data, self.authors_labels, self.titles_labels = dataset.data_for_kfold()
        # divide the dataset in train/test
        self.x_tr, self.x_te, self.y_tr, self.y_te = \
            train_test_split(self.data, self.authors_labels, test_size=0.2, random_state=42, stratify=self.authors_labels)
        print(f'#training samples = {len(self.y_tr)}')
        print(f'#test samples = {len(self.y_te)}')

        # sort the sets by lengths (for padding)
        #self.x_tr, self.y_tr = self._sort_docs(x_tr, y_tr)
        #self.x_te, self.y_te = self._sort_docs(x_te, y_te)

        # create the analyzer for each encoding: a CountVectorizer based on training samples
        print('----- CREATING ANALYZERS -----')
        # sillabic quantities
        syll_tr = metric_scansion(self.x_tr)
        self.anal_syll = CountVectorizer(analyzer='char', ngram_range=(1,1)).fit(syll_tr)
        print(f'Syll analyzer [Done]')
        # DVSA distortion
        DVSA_tr = self._DVSA(self.x_tr)
        self.anal_DVSA = CountVectorizer(analyzer='word', token_pattern=r'[^\s]+', min_df=3).fit(DVSA_tr)
        print(f'DVSA analyzer [Done]')
        # DVEX distortion
        DVEX_tr = self._DVEX(self.x_tr)
        self.anal_DVEX = CountVectorizer(analyzer='char', ngram_range=(1,1)).fit(DVEX_tr)
        print(f'DVEX analyzer [Done]')

        # create the training generator (for batches)
        training_dataset = NN_BaseDataset(self.x_tr, self.y_tr)
        self.training_generator = DataLoader(training_dataset, batch_size, shuffle=True,
                                             num_workers=2, collate_fn=self._collate_padding)
        # adding 2 for unknown and padding
        self.vocab_lens = [len(self.anal_syll.vocabulary_) + 2, len(self.anal_DVSA.vocabulary_) + 2,
                           len(self.anal_DVEX.vocabulary_) + 2]

    # DV-EX text distortion method from Stamatatos_2018:
    # Every word not in latin_function_words is masked by replacing each of its INTERIOR characters with an asterisk (*).
    # for character embedding
    def _DVEX(self, docs):
        dis_texts = []
        for doc in docs:
            mod_tokens = tokenize(doc)
            dis_text = ''
            for token in mod_tokens:
                dis_token = ''
                if dis_text != '':  # if it's not the first token, put a space
                    dis_text += ' '
                if token in self.function_words or len(token) == 1:
                    dis_token = token  # if it's a function word, or it's a one char token, give it like that
                else:
                    dis_token = token[0] + ('*' * (len(token) - 2)) + token[len(token) - 1]  # otherwise, distortion
                dis_text += dis_token
            dis_texts.append(dis_text)
        return dis_texts

    # DV-SA text distortion method from Stamatatos_2018:
    # Every word not in latin_function_words is replaced by an asterisk (*).
    # for word embedding
    def _DVSA(self, docs):
        dis_texts = []
        for doc in docs:
            mod_tokens = tokenize(doc)
            dis_text = ''
            for token in mod_tokens:
                dis_token = ''
                if dis_text != '':  # if it's not the first token, put a space
                    dis_text += ' '
                if token in self.function_words:
                    dis_token = token  # if it's a function word, or it's a one char token, give it like that
                else:
                    dis_token = '*'  # otherwise, distortion
                dis_text += dis_token
            dis_texts.append(dis_text)
        return dis_texts

    # encode the texts based on the analyzer (>>indices)
    # value for unknown word: len of vocabulary (== max idx +1)
    def _encode(self, docs, analyzer):
        encoded_texts = []
        vocab = analyzer.vocabulary_
        tokenizer = analyzer.build_analyzer()
        for doc in docs:
            encoded_text = [vocab.get(item, len(vocab)) for item in tokenizer(doc)]
            encoded_texts.append(encoded_text)
        return encoded_texts

    # perform all three encodings
    def _encoding_all(self, data):
        data_syll = self._encode(metric_scansion(data), self.anal_syll)
        data_DVSA = self._encode(self._DVSA(data), self.anal_DVSA)
        data_DVEX = self._encode(self._DVEX(data), self.anal_DVEX)
        return data_syll, data_DVSA, data_DVEX

    #sort the documents (with labels) based on length
    def _sort_docs(self, docs, labels):
        tuples = list(zip(docs, labels))
        tuples = sorted(tuples, key=lambda x: len(x[0]))
        return list(zip(*tuples))

    #transform each label into a 1-hot-tensor and stack them together
    def _hot_tensor(self, labels):
        hot_tens = []
        for label in labels:
            hot_ten = torch.zeros(len(self.authors))
            hot_ten[label] = 1
            hot_tens.append(hot_ten)
        return torch.stack(hot_tens)

    def _collate_padding(self, batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        data_syll, data_DVSA, data_DVEX = self._encoding_all(data)
        pad_syll = pad_sequence([torch.Tensor(doc) for doc in data_syll], batch_first=True,
                                padding_value=self.vocab_lens[0] - 1)
        pad_DVSA = pad_sequence([torch.Tensor(doc) for doc in data_DVSA], batch_first=True,
                                padding_value=self.vocab_lens[1] - 1)
        pad_DVEX = pad_sequence([torch.Tensor(doc) for doc in data_DVEX], batch_first=True,
                                padding_value=self.vocab_lens[2] - 1)
        targets = torch.Tensor(labels)
        return [pad_syll.long(), pad_DVSA.long(), pad_DVEX.long(), targets.long()]

    def __max_len(self, docs):
        return len(max(docs, key=len))

    def __padding(self, docs, pad_value):
        return [item + [pad_value] * (self.__max_len(docs) - len(item)) for item in docs]
