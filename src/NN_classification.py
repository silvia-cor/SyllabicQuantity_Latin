from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from general.helpers import data_for_kfold
from dataset_prep.NN_dataloader import NN_DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Penta_NN(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, kernel_sizes=[3,5,7]):
        super().__init__()
        assert NN_params['fake'] != NN_params['SQ'], 'Must have either fake or SQ channel, but not both'
        if NN_params['fake']:
            self.embed_fake = nn.Embedding(vocab_lens['fake'], 16)
            self.cnn_fake = nn.ModuleList([nn.Conv1d(16, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_fake = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        if NN_params['SQ']:
            self.embed_SQ = nn.Embedding(vocab_lens['SQ'], 16)
            self.cnn_SQ = nn.ModuleList([nn.Conv1d(16, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_SQ = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        if NN_params['DVMA']:
            self.embed_DVMA= nn.Embedding(vocab_lens['DVMA'], 32)
            self.cnn_DVMA = nn.ModuleList([nn.Conv1d(32, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_DVMA = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        if NN_params['DVSA']:
            self.embed_DVSA= nn.Embedding(vocab_lens['DVSA'], 100)
            self.cnn_DVSA = nn.ModuleList([nn.Conv1d(100, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_DVSA =nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        if NN_params['DVEX']:
            self.embed_DVEX= nn.Embedding(vocab_lens['DVEX'], 32)
            self.cnn_DVEX = nn.ModuleList([nn.Conv1d(32, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_DVEX = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        if NN_params['DVL2']:
            self.embed_DVL2= nn.Embedding(vocab_lens['DVL2'], 32)
            self.cnn_DVL2 = nn.ModuleList([nn.Conv1d(32, 100, kernel_size) for kernel_size in kernel_sizes])
            self.lstm_DVL2 = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_labels)

    def forward(self, enc_SQ, enc_DVMA, enc_DVSA, enc_DVEX, enc_DVL2, batch_size):
        #channel 1
        hid1 = self.embed1(enc_syll)
        hid1 = [conv_block(hid1, conv) for conv in self.cnn1]
        hid1 = torch.cat(hid1, 1)
        out1 = lstm_block(hid1, self.lstm1, batch_size)
        #out1 = self.flat(hid1)
        # channel 2
        hid2 = self.embed2(enc_DVSA)
        hid2 = conv_block(hid2, self.cnn2)
        out2 = lstm_block(hid2, self.lstm2, batch_size)
        #out2 = self.flat(hid2)
        # channel 3
        hid3 = self.embed3(enc_DVEX)
        hid3 = conv_block(hid3, self.cnn3)
        out3 = lstm_block(hid3, self.lstm3, batch_size)
        #out3 = self.flat(hid3)

        merge_out = torch.cat((out1, out2, out3), 1)
        drop_out = self.drop(merge_out)
        final_out = self.dense(F.relu(drop_out))
        return final_out



def NN_classification(dataset, NN_params, kfold, n_sent):
    assert isinstance(kfold, StratifiedKFold), 'Only kfold CV implemented for NN'
    authors, titles, data, authors_labels, titles_labels = data_for_kfold(dataset)

    for i, (train_index, test_index) in enumerate(kfold.split(data, authors_labels)):
        print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
        x_tr = data[train_index]
        x_te = data[test_index]
        y_tr = authors_labels[train_index]
        y_te = authors_labels[test_index]

        dataset = NN_DataLoader(x_tr, x_te, y_tr, y_te, n_sent, batch_size=128)
