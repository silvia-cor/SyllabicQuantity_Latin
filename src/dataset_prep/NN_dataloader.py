from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from general.helpers import get_function_words, make_fake_vocab, fake_metric_scansion, metric_scansion, dis_DVMA, dis_DVEX, dis_DVSA, dis_DVL2, tokenize_nopunct
import nltk
from sklearn.preprocessing import normalize


# pytorch Dataset class (for DataLoader)
class NN_BaseDataset(Dataset):
    def __init__(self, x, x_cltk, y):
        self.x = x
        self.x_cltk = x_cltk
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.x_cltk[index], self.y[index]


# main class: create and divides (train/test, batches) the dataset_prep
class NN_DataLoader:
    def __init__(self, data, data_cltk, authors_labels, NN_params, batch_size):
        print('----- CREATING DATASET -----')
        self.function_words = get_function_words('latin') # for distortion techniques
        self.vocab_lens = {}
        self.NN_params = NN_params
        # devide the dataset into train+val and test
        x_trval, x_te, x_trval_cltk, x_te_cltk, y_trval, y_te = train_test_split(data, data_cltk, authors_labels, test_size=0.1, random_state=42, stratify=authors_labels)
        # divide the train+val so that the dataset is train/val/test
        x_tr, x_val, x_tr_cltk, x_val_cltk, y_tr, y_val = train_test_split(x_trval, x_trval_cltk, y_trval, test_size=0.1, random_state=42, stratify=y_trval)

        print(f'#training samples = {len(y_tr)}')
        print(f'#validation samples = {len(y_val)}')
        print(f'#test samples = {len(y_te)}')

        # sort the sets by lengths (for padding)
        # self.x_tr, self.y_tr = self._sort_docs(x_tr, y_tr)
        # self.x_te, self.y_te = self._sort_docs(x_te, y_te)

        # create the analyzer for each encoding: a CountVectorizer based on training samples
        print('----- CREATING ANALYZERS -----')
        if self.NN_params['FAKE']:
            # fake syllabic quantities
            # Make a word-based CountVectorizer based on the entire dataset
            anal_words = CountVectorizer(analyzer='word', ngram_range=(1, 1)).fit(np.concatenate((x_trval, x_te)))
            print(f'Word analyzer [Done]')
            # Transform the vocabulary so that each word corresponds to a random SQ sequence
            self.fake_vocab = make_fake_vocab(anal_words)
            # create the fake-SQ CountVectorizer by replacing the words to the fake SQ sequence
            x_dis = fake_metric_scansion(x_tr, self.fake_vocab)
            self.anal_FAKE = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['FAKE'] = len(self.anal_FAKE.vocabulary_)
            print(f'FAKE analyzer [Done]')
        if self.NN_params['SQ']:
            # sillabic quantities
            x_dis = x_tr_cltk
            self.anal_SQ = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['SQ'] = len(self.anal_SQ.vocabulary_)
            print(f'SQ analyzer [Done]')
        if self.NN_params['DVMA']:
            # DVMA distortion
            x_dis = dis_DVMA(x_tr, self.function_words)
            self.anal_DVMA = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['DVMA'] = len(self.anal_DVMA.vocabulary_)
            print(f'DVMA analyzer [Done]')
        if self.NN_params['DVSA']:
            # DVSA distortion
            x_dis = dis_DVSA(x_tr, self.function_words)
            self.anal_DVSA = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['DVSA'] = len(self.anal_DVSA.vocabulary_)
            print(f'DVSA analyzer [Done]')
        if self.NN_params['DVEX']:
            # DVEX distortion
            x_dis = dis_DVEX(x_tr, self.function_words)
            self.anal_DVEX = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['DVEX'] = len(self.anal_DVEX.vocabulary_)
            print(f'DVEX analyzer [Done]')
        if self.NN_params['DVL2']:
            # DVL2 distortion
            x_dis = dis_DVL2(x_tr, self.function_words)
            self.anal_DVL2 = self._make_analyzer(CountVectorizer(analyzer='char', ngram_range=(1, 1)), x_dis)
            self.vocab_lens['DVL2'] = len(self.anal_DVL2.vocabulary_)
            print(f'DVL2 analyzer [Done]')

        # create the train/val/test generator (for batches)
        train_dataset = NN_BaseDataset(x_tr, x_tr_cltk, y_tr)
        self.train_generator = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=5, collate_fn=self._collate_padding, worker_init_fn=self._seed_worker)
        val_dataset = NN_BaseDataset(x_val, x_val_cltk, y_val)
        self.val_generator = DataLoader(val_dataset, batch_size, num_workers=5, collate_fn=self._collate_padding, worker_init_fn=self._seed_worker)
        test_dataset = NN_BaseDataset(x_te, x_te_cltk, y_te)
        self.test_generator = DataLoader(test_dataset, batch_size, num_workers=5, collate_fn=self._collate_padding, worker_init_fn=self._seed_worker)

    def _collate_padding(self, batch):
        data = [item[0] for item in batch]
        data_cltk = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        encodings = {}
        if self.NN_params['FAKE']:
            dis_data = self._encode(fake_metric_scansion(data, self.fake_vocab), self.anal_FAKE)
            encodings['FAKE'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_FAKE.vocabulary_['<pad>']).long()
        if self.NN_params['SQ']:
            dis_data = self._encode(data_cltk, self.anal_SQ)
            encodings['SQ'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_SQ.vocabulary_['<pad>']).long()
        if self.NN_params['DVMA']:
            dis_data = self._encode(dis_DVMA(data, self.function_words), self.anal_DVMA)
            encodings['DVMA'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_DVMA.vocabulary_['<pad>']).long()
        if self.NN_params['DVSA']:
            dis_data = self._encode(dis_DVSA(data, self.function_words), self.anal_DVSA)
            encodings['DVSA'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_DVSA.vocabulary_['<pad>']).long()
        if self.NN_params['DVEX']:
            dis_data = self._encode(dis_DVEX(data, self.function_words), self.anal_DVEX)
            encodings['DVEX'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_DVEX.vocabulary_['<pad>']).long()
        if self.NN_params['DVL2']:
            dis_data = self._encode(dis_DVL2(data, self.function_words), self.anal_DVL2)
            encodings['DVL2'] = pad_sequence([torch.Tensor(doc) for doc in dis_data], batch_first=True, padding_value=self.anal_DVL2.vocabulary_['<pad>']).long()
        targets = torch.Tensor(labels).long()
        feats = _features(data, self.function_words)
        return encodings, targets, feats

    # set the seed for the DataLoader worker
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _make_analyzer(self, vectorizer, docs):
        analyzer = vectorizer.fit(docs)
        analyzer.vocabulary_['<unk>'] = len(analyzer.vocabulary_)
        analyzer.vocabulary_['<pad>'] = len(analyzer.vocabulary_)
        return analyzer

    # encode the texts based on the analyzer (>>indices)
    def _encode(self, docs, analyzer):
        encoded_texts = []
        vocab = analyzer.vocabulary_
        tokenizer = analyzer.build_analyzer()
        for doc in docs:
            encoded_text = [vocab.get(item, vocab.get('<unk>')) for item in tokenizer(doc)]
            encoded_texts.append(encoded_text)
        return encoded_texts


def _features(documents, function_words, upto=26, min=1, max=101):
    features_all = []
    for text in documents:
        feats = []
        mod_tokens = tokenize_nopunct(text)
        freqs = nltk.FreqDist(mod_tokens)
        nwords = len(mod_tokens)
        funct_words_freq = [freqs[function_word] / nwords for function_word in function_words]
        feats.extend(funct_words_freq)
        tokens_len = [len(token) for token in mod_tokens]
        tokens_count = []
        for i in range(1, upto):
            tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
        feats.extend(tokens_count)
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        nsent = len(sentences)
        sent_len = []
        sent_count = []
        for sentence in sentences:
            mod_tokens = tokenize_nopunct(sentence)
            sent_len.append(len(mod_tokens))
        for i in range(min, max):
            sent_count.append((sum(j >= i for j in sent_len)) / nsent)
        feats.extend(sent_count)
        features_all.append(feats)
    return torch.Tensor(normalize(features_all))
