import torch
import torch.nn as nn
import torch.nn.functional as F


class Penta_Ensemble(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, start_dim_dense, device, kernel_sizes=[3,4,5,6,7]):
        super().__init__()
        self.emb_len = 32
        self.cnn_out_size = 100
        self.n_labels = n_labels
        self.device =device
        if NN_params['FAKE']:
            self.embed_FAKE, self.cnn_FAKE1, self.cnn_FAKE2, self.cnn_FAKE3, self.dense_FAKE1, self.dense_FAKE2 = self.make_layers(vocab_lens['FAKE'], kernel_sizes)
        if NN_params['SQ']:
            self.embed_SQ, self.cnn_SQ1, self.cnn_SQ2, self.cnn_SQ3, self.dense_SQ1, self.dense_SQ2 = self.make_layers(vocab_lens['SQ'], kernel_sizes)
        if NN_params['DVMA']:
            self.embed_DVMA, self.cnn_DVMA1, self.cnn_DVMA2, self.cnn_DVMA3, self.dense_DVMA1, self.dense_DVMA2 = self.make_layers(vocab_lens['DVMA'], [3])
        if NN_params['DVSA']:
            self.embed_DVSA, self.cnn_DVSA1, self.cnn_DVSA2, self.cnn_DVSA3, self.dense_DVSA1, self.dense_DVSA2 = self.make_layers(vocab_lens['DVSA'], [3])
        if NN_params['DVEX']:
            self.embed_DVEX, self.cnn_DVEX1, self.cnn_DVEX2, self.cnn_DVEX3, self.dense_DVEX1, self.dense_DVEX2 = self.make_layers(vocab_lens['DVEX'], [3])
        if NN_params['DVL2']:
            self.embed_DVL2, self.cnn_DVL21, self.cnn_DVL22, self.cnn_DVL23, self.dense_DVL21, self.dense_DVL22 = self.make_layers(vocab_lens['DVL2'], [3])
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(3)
        self.final = nn.Linear(self.n_labels, self.n_labels)

    def forward(self, NN_params, encodings): #, feats):
        outputs = []
        if NN_params['FAKE']:
            conv_stack = [list(convs) for convs in zip(self.cnn_FAKE1, self.cnn_FAKE2, self.cnn_FAKE3)]
            outputs.append(self.sub_forward(encodings['FAKE'], self.embed_FAKE, conv_stack, self.dense_FAKE1, self.dense_FAKE2).unsqueeze(dim=2))
        if NN_params['SQ']:
            conv_stack = [list(convs) for convs in zip(self.cnn_SQ1, self.cnn_SQ2, self.cnn_SQ3)]
            outputs.append(self.sub_forward(encodings['SQ'], self.embed_SQ, conv_stack, self.dense_SQ1, self.dense_SQ2).unsqueeze(dim=2))
        if NN_params['DVMA']:
            outputs.append(self.sub_forward(encodings['DVMA'], self.embed_DVMA, [self.cnn_DVMA1, self.cnn_DVMA2, self.cnn_DVMA3],
                                            self.dense_DVMA1, self.dense_DVMA2).unsqueeze(dim=2))
        if NN_params['DVSA']:
            outputs.append(self.sub_forward(encodings['DVSA'], self.embed_DVSA, [self.cnn_DVSA1, self.cnn_DVSA2, self.cnn_DVSA3],
                                            self.dense_DVSA1, self.dense_DVSA2).unsqueeze(dim=2))
        if NN_params['DVEX']:
            outputs.append(self.sub_forward(encodings['DVEX'], self.embed_DVEX, [self.cnn_DVEX1, self.cnn_DVEX2, self.cnn_DVEX3],
                                            self.dense_DVEX1, self.dense_DVEX2).unsqueeze(dim=2))
        if NN_params['DVL2']:
            outputs.append(self.sub_forward(encodings['DVL2'], self.embed_DVL2, [self.cnn_DVL21, self.cnn_DVL22, self.cnn_DVL23],
                                            self.dense_DVL21, self.dense_DVL22).unsqueeze(dim=2))
        x = torch.cat(outputs, dim=2)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.squeeze(2)
        #x = self.final(x)
        return x

    def conv_block(self, input, conv_layers):   # input is (N, L, Cin)
        # input has shape (Batch-size, Length of doc, Embedding-size) in documentation (N, L, Cin)
        x = input.transpose(1, 2).contiguous()  # (N, Cin, L)
        for i, conv_layer in enumerate(conv_layers):
            x = conv_layer(x)  # (N, Cout, L)
            x = F.relu(x)  # (N, Cout, L)
            if i < len(conv_layers)-1:
                x = self.maxpool(x)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x  # output (N, Cout)

    def make_layers(self, vocab_len, kernel_sizes):
        embed = nn.Embedding(vocab_len, self.emb_len)
        if len(kernel_sizes) > 1:
            cnn1 = nn.ModuleList([nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
            cnn2 = nn.ModuleList([nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
            cnn3 = nn.ModuleList([nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
        else:
            cnn1 = nn.Conv1d(self.emb_len, self.cnn_out_size, kernel_sizes[0])
            cnn2 = nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_sizes[0])
            cnn3 = nn.Conv1d(self.cnn_out_size, self.cnn_out_size, kernel_sizes[0])
        dense1 = nn.Linear(self.cnn_out_size * len(kernel_sizes), 516)
        dense2 = nn.Linear(516, self.n_labels)
        return embed, cnn1, cnn2, cnn3, dense1, dense2

    def sub_forward(self, encoding, embed_layer, conv_stack, dense_layer1, dense_layer2):
        x = embed_layer(encoding.to(self.device))
        if all(isinstance(conv_group, list) for conv_group in conv_stack):
            x = [self.conv_block(x, conv_group) for conv_group in conv_stack]
            x = torch.cat(x, 1)
        else:
            x = self.conv_block(x, conv_stack)
        x = self.flat(x)
        x = self.drop(x)
        x = dense_layer1(F.relu(x))
        x = self.drop(x)
        x = dense_layer2(F.relu(x))
        return x
