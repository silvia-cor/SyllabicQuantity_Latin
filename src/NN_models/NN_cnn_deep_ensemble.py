import torch
import torch.nn as nn
import torch.nn.functional as F


class Penta_DeepEnsemble(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, feat_dim, device):
        super().__init__()
        self.emb_len = 32
        self.cnn_out_size = 128
        self.kernel_sizes = [3, 5]
        self.dense_size = 256
        self.n_labels = n_labels
        self.device = device
        self.feat_dim = feat_dim
        self.dense_feat1 = nn.Linear(self.feat_dim, self.dense_size)
        self.dense_feat2 = nn.Linear(self.dense_size, self.n_labels)
        if NN_params['FAKE']:
            self.embed_FAKE, self.cnn_FAKE1, self.cnn_FAKE2, self.dense_FAKE = self.make_layers(vocab_lens['FAKE'],
                                                                                                self.kernel_sizes)
        if NN_params['SQ']:
            self.embed_SQ, self.cnn_SQ1, self.cnn_SQ2, self.dense_SQ = self.make_layers(vocab_lens['SQ'],
                                                                                        self.kernel_sizes)
        if NN_params['DVMA']:
            self.embed_DVMA, self.cnn_DVMA1, self.cnn_DVMA2, self.dense_DVMA = self.make_layers(vocab_lens['DVMA'],
                                                                                                self.kernel_sizes)
        if NN_params['DVSA']:
            self.embed_DVSA, self.cnn_DVSA1, self.cnn_DVSA2, self.dense_DVSA = self.make_layers(vocab_lens['DVSA'],
                                                                                                self.kernel_sizes)
        if NN_params['DVEX']:
            self.embed_DVEX, self.cnn_DVEX1, self.cnn_DVEX2, self.dense_DVEX = self.make_layers(vocab_lens['DVEX'],
                                                                                                self.kernel_sizes)
        if NN_params['DVL2']:
            self.embed_DVL2, self.cnn_DVL21, self.cnn_DVL22, self.dense_DVL2 = self.make_layers(vocab_lens['DVL2'],
                                                                                                self.kernel_sizes)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(3)
        self.final = nn.Linear(self.n_labels, self.n_labels)

    def forward(self, NN_params, encodings, feats):
        outputs = []
        f = F.relu(self.dense_feat1(feats))
        f = self.drop(f)
        f = F.relu(self.dense_feat2(f))
        outputs.append(f.unsqueeze(dim=2))
        if NN_params['FAKE']:
            conv_stack = [list(convs) for convs in zip(self.cnn_FAKE1, self.cnn_FAKE2)]
            outputs.append(
                self.sub_forward(encodings['FAKE'], self.embed_FAKE, conv_stack, self.dense_FAKE1).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['FAKE'], self.embed_FAKE, [self.cnn_FAKE1, self.cnn_FAKE2],
            #                                 self.dense_FAKE).unsqueeze(dim=2))
        if NN_params['SQ']:
            conv_stack = [list(convs) for convs in zip(self.cnn_SQ1, self.cnn_SQ2)]
            outputs.append(
                self.sub_forward(encodings['SQ'], self.embed_SQ, conv_stack, self.dense_SQ).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['SQ'], self.embed_SQ, [self.cnn_SQ1, self.cnn_SQ2],
            #                                self.dense_SQ).unsqueeze(dim=2))
        if NN_params['DVMA']:
            conv_stack = [list(convs) for convs in zip(self.cnn_DVMA1, self.cnn_DVMA2)]
            outputs.append(
                self.sub_forward(encodings['DVMA'], self.embed_DVMA, conv_stack, self.dense_DVMA).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['DVMA'], self.embed_DVMA, [self.cnn_DVMA1, self.cnn_DVMA2],
            #                                self.dense_DVMA).unsqueeze(dim=2))
        if NN_params['DVSA']:
            conv_stack = [list(convs) for convs in zip(self.cnn_DVSA1, self.cnn_DVSA2)]
            outputs.append(
                self.sub_forward(encodings['DVSA'], self.embed_DVSA, conv_stack, self.dense_DVSA).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['DVSA'], self.embed_DVSA, [self.cnn_DVSA1, self.cnn_DVSA2],
            #                                self.dense_DVSA).unsqueeze(dim=2))
        if NN_params['DVEX']:
            conv_stack = [list(convs) for convs in zip(self.cnn_DVEX1, self.cnn_DVEX2)]
            outputs.append(
                self.sub_forward(encodings['DVEX'], self.embed_DVEX, conv_stack, self.dense_DVEX).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['DVEX'], self.embed_DVEX, [self.cnn_DVEX1, self.cnn_DVEX2],
            #                                self.dense_DVEX).unsqueeze(dim=2))
        if NN_params['DVL2']:
            conv_stack = [list(convs) for convs in zip(self.cnn_DVL21, self.cnn_DVL22)]
            outputs.append(
                self.sub_forward(encodings['DVL2'], self.embed_DVL2, conv_stack, self.dense_DVL2).unsqueeze(dim=2))
            # outputs.append(self.sub_forward(encodings['DVL2'], self.embed_DVL2, [self.cnn_DVL21, self.cnn_DVL22],
            #                                self.dense_DVL2).unsqueeze(dim=2))
        x = torch.cat(outputs, dim=2)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        x = self.final(x)
        return x

    def conv_block(self, input, conv_layers):  # input is (N, L, Cin)
        # input has shape (Batch-size, Length of doc, Embedding-size) in documentation (N, L, Cin)
        x = input.transpose(1, 2).contiguous()  # (N, Cin, L)
        for i, conv_layer in enumerate(conv_layers):
            x = conv_layer(x)  # (N, Cout, L)
            x = F.relu(x)  # (N, Cout, L)
            if i < len(conv_layers) - 1:
                x = self.maxpool(x)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x  # output (N, Cout)

    def make_layers(self, vocab_len, kernel_sizes):
        embed = nn.Embedding(vocab_len, self.emb_len)
        if isinstance(kernel_sizes, list):
            cnn1 = nn.ModuleList(
                [nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
            cnn2 = nn.ModuleList(
                [nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
        else:
            cnn1 = nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_sizes)
            cnn2 = nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_sizes)
        dense = nn.Linear(self.cnn_out_size, self.dense_size)
        return embed, cnn1, cnn2, dense

    def sub_forward(self, encoding, embed_layer, conv_stack, dense_layer):
        x = embed_layer(encoding.to(self.device))
        if all(isinstance(conv_group, list) for conv_group in conv_stack):
            x = [self.conv_block(x, conv_group) for conv_group in conv_stack]
            x = torch.cat(x, 1)
        else:
            x = self.conv_block(x, conv_stack)
        x = self.flat(x)
        x = self.drop(x)
        x = F.relu(dense_layer(x))
        return x
