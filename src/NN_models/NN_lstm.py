import torch
import torch.nn as nn
import torch.nn.functional as F


class Penta_Lstm(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, start_dim_dense, device):
        super().__init__()
        dim_dense = start_dim_dense
        self.emb_len = 32
        self.cnn_out_size = 100
        self.lstm_hid_size = 100
        self.lstm_num_layers = 3
        self.lstm_bidirect = True
        self.device = device
        if NN_params['FAKE']:
            self.embed_FAKE, self.cnn_FAKE, self.lstm_FAKE = self.make_layers(vocab_lens['FAKE'])
            dim_dense += self.lstm_hid_size
        if NN_params['SQ']:
            self.embed_SQ, self.cnn_SQ, self.lstm_SQ = self.make_layers(vocab_lens['SQ'])
            dim_dense += self.lstm_hid_size
        if NN_params['DVMA']:
            self.embed_DVMA, self.cnn_DVMA, self.lstm_DVMA = self.make_layers(vocab_lens['DVMA'])
            dim_dense += self.lstm_hid_size
        if NN_params['DVSA']:
            self.embed_DVSA, self.cnn_DVSA, self.lstm_DVSA = self.make_layers(vocab_lens['DVSA'])
            dim_dense += self.lstm_hid_size
        if NN_params['DVEX']:
            self.embed_DVEX, self.cnn_DVEX, self.lstm_DVEX = self.make_layers(vocab_lens['DVEX'])
            dim_dense += self.lstm_hid_size
        if NN_params['DVL2']:
            self.embed_DVL2, self.cnn_DVL2, self.lstm_DVL2 = self.make_layers(vocab_lens['DVL2'])
            dim_dense += self.lstm_hid_size
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(3)
        self.dense1 = nn.Linear(dim_dense, 256)
        self.dense2 = nn.Linear(256, n_labels)

    def forward(self, NN_params, encodings): #, feats):
        outputs = []
        if NN_params['FAKE']:
            outputs.append(self.sub_forward(encodings['FAKE'], self.embed_FAKE, self.cnn_FAKE, self.lstm_FAKE))
        if NN_params['SQ']:
            outputs.append(self.sub_forward(encodings['SQ'], self.embed_SQ, self.cnn_SQ, self.lstm_SQ))
        if NN_params['DVMA']:
            outputs.append(self.sub_forward(encodings['DVMA'], self.embed_DVMA, self.cnn_DVMA, self.lstm_DVMA))
        if NN_params['DVSA']:
            outputs.append(self.sub_forward(encodings['DVSA'], self.embed_DVSA, self.cnn_DVSA, self.lstm_DVSA))
        if NN_params['DVEX']:
            outputs.append(self.sub_forward(encodings['DVEX'], self.embed_DVEX, self.cnn_DVEX, self.lstm_DVEX))
        if NN_params['DVL2']:
            outputs.append(self.sub_forward(encodings['DVL2'], self.embed_DVL2, self.cnn_DVL2, self.lstm_DVL2))
        x = torch.cat(outputs, 1)
        # x = torch.cat((x, feats), 1)
        x = self.drop(x)
        x = self.dense1(F.relu(x))
        x = self.drop(x)
        x = self.dense2(F.relu(x))
        return x

    def make_layers(self, vocab_len):
        embed = nn.Embedding(vocab_len, self.emb_len)
        cnn = nn.Conv1d(in_channels=self.emb_len, out_channels=self.cnn_out_size, kernel_size=3)
        lstm = nn.LSTM(input_size=self.cnn_out_size, hidden_size=self.lstm_hid_size, num_layers=self.lstm_num_layers,
                       bidirectional=self.lstm_bidirect, batch_first=True)
        return embed, cnn, lstm

    def sub_forward(self, encoding, embed_layer, cnn_layer, lstm_layer):
        x = embed_layer(encoding.to(self.device))
        x = x.transpose(1, 2).contiguous()
        x = cnn_layer(x)
        x = x.transpose(1, 2).contiguous()
        lstm_out, (ht, ct) = lstm_layer(x)
        x = ht[-1]
        return x
