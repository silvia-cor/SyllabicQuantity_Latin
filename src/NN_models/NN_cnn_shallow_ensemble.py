import torch
import torch.nn as nn
import torch.nn.functional as F


class Penta_ShallowEnsemble(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, device):
        super().__init__()
        self.emb_len = 32
        self.kernel_size = [3]
        self.kernel_sizes = [3, 4, 5]
        self.cnn_out_size = 256
        self.feat_dim = 205
        self.dense_size = 512
        self.n_labels = n_labels
        self.device = device
        if NN_params['FAKE']:
            self.embed_FAKE, self.cnn_FAKE, self.dense_FAKE = self.make_layers(vocab_lens['FAKE'], self.kernel_sizes)
        if NN_params['SQ']:
            self.embed_SQ, self.cnn_SQ, self.dense_SQ = self.make_layers(vocab_lens['SQ'], self.kernel_sizes)
        if NN_params['DVMA']:
            self.embed_DVMA, self.cnn_DVMA, self.dense_DVMA = self.make_layers(vocab_lens['DVMA'], self.kernel_size)
        if NN_params['DVSA']:
            self.embed_DVSA, self.cnn_DVSA, self.dense_DVSA = self.make_layers(vocab_lens['DVSA'], self.kernel_size)
        if NN_params['DVEX']:
            self.embed_DVEX, self.cnn_DVEX, self.dense_DVEX = self.make_layers(vocab_lens['DVEX'], self.kernel_size)
        if NN_params['DVL2']:
            self.embed_DVL2, self.cnn_DVL2, self.dense_DVL2 = self.make_layers(vocab_lens['DVL2'], self.kernel_size)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(3)
        self.feat_dense1 = nn.Linear(self.feat_dim, self.dense_size)
        self.feat_dense2 = nn.Linear(self.dense_size, self.n_labels)
        self.final = nn.Linear(self.n_labels, self.n_labels)

    def forward(self, NN_params, encodings, feats):
        outputs = []
        f = self.feat_dense1(F.relu(feats))
        f = self.drop(f)
        f = self.feat_dense2(F.relu(f))
        outputs.append(f.unsqueeze(dim=2))
        if NN_params['FAKE']:
            outputs.append(self.sub_forward(encodings['FAKE'], self.embed_FAKE, self.cnn_FAKE, self.dense_FAKE).unsqueeze(dim=2))
        if NN_params['SQ']:
            outputs.append(self.sub_forward(encodings['SQ'], self.embed_SQ, self.cnn_SQ, self.dense_SQ).unsqueeze(dim=2))
        if NN_params['DVMA']:
            outputs.append(self.sub_forward(encodings['DVMA'], self.embed_DVMA, self.cnn_DVMA, self.dense_DVMA).unsqueeze(dim=2))
        if NN_params['DVSA']:
            outputs.append(self.sub_forward(encodings['DVSA'], self.embed_DVSA, self.cnn_DVSA, self.dense_DVSA).unsqueeze(dim=2))
        if NN_params['DVEX']:
            outputs.append(self.sub_forward(encodings['DVEX'], self.embed_DVEX, self.cnn_DVEX, self.dense_DVEX).unsqueeze(dim=2))
        if NN_params['DVL2']:
            outputs.append(self.sub_forward(encodings['DVL2'], self.embed_DVL2, self.cnn_DVL2, self.dense_DVL2).unsqueeze(dim=2))
        x = torch.cat(outputs, dim=2)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = self.final(x)
        return x

    def conv_block(self, input, conv_layer):   # input is (N, L, Cin)
        # input has shape (Batch-size, Length of doc, Embedding-size) in documentation (N, L, Cin)
        x = input.transpose(1, 2).contiguous()  # (N, Cin, L)
        x = conv_layer(x)  # (N, Cout, L)
        x = F.relu(x)  # (N, Cout, L)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x  # output (N, Cout)

    def make_layers(self, vocab_len, kernel_sizes):
        embed = nn.Embedding(vocab_len, self.emb_len)
        # cnn = nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_size=self.kernel_size)
        if len(kernel_sizes) > 1:
            cnn = nn.ModuleList([nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
        else:
            cnn = nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_sizes[0])
        dense = nn.Linear(self.cnn_out_size, self.n_labels)
        return embed, cnn, dense

    def sub_forward(self, encoding, embed_layer, cnn_layer, dense_layer):
        x = embed_layer(encoding.to(self.device))
        # x = self.conv_block(x, cnn_layer)
        if isinstance(cnn_layer, nn.ModuleList):
            x = [self.conv_block(x, conv_kernel) for conv_kernel in cnn_layer]
            # x = torch.cat(x, dim=2)
            # L = x.size()[2]
            # x = F.max_pool1d(x, L)
            # x = x.squeeze(2)
            x = torch.cat(x, 1)
        else:
            x = self.conv_block(x, cnn_layer)
        x = self.flat(x)
        x = self.drop(x)
        x = dense_layer(F.relu(x))
        return x
