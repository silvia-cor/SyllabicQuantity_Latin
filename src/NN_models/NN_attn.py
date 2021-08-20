import torch
import torch.nn as nn
import torch.nn.functional as F


class Penta_Attn(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, start_dim_dense, device):
        super().__init__()
        dim_dense = start_dim_dense
        self.emb_len = 16
        self.n_heads = 1
        self.num_layers = 2
        self.cnn_size = 50
        self.device = device
        if NN_params['FAKE']:
            self.embed_FAKE, self.attn_FAKE, self.cnn_FAKE = self.make_layers(vocab_lens['FAKE'])
            dim_dense += self.cnn_size
        if NN_params['SQ']:
            self.embed_SQ, self.attn_SQ, self.cnn_SQ = self.make_layers(vocab_lens['SQ'])
            dim_dense += self.cnn_size
        if NN_params['DVMA']:
            self.embed_DVMA, self.attn_DVMA, self.cnn_DVMA = self.make_layers(vocab_lens['DVMA'])
            dim_dense += self.cnn_size
        if NN_params['DVSA']:
            self.embed_DVSA, self.attn_DVSA, self.cnn_DVSA = self.make_layers(vocab_lens['DVSA'])
            dim_dense += self.cnn_size
        if NN_params['DVEX']:
            self.embed_DVEX, self.attn_DVEX, self.cnn_DVEX = self.make_layers(vocab_lens['DVEX'])
            dim_dense += self.cnn_size
        if NN_params['DVL2']:
            self.embed_DVL2, self.attn_DVL2, self.cnn_DVL2 = self.make_layers(vocab_lens['DVL2'])
            dim_dense += self.cnn_size
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.dense1 = nn.Linear(dim_dense, n_labels)

    def forward(self, NN_params, encodings): #, feats):
        outputs = []
        if NN_params['FAKE']:
            outputs.append(self.sub_forward(encodings['FAKE'], self.embed_FAKE, self.attn_FAKE, self.cnn_FAKE))
        if NN_params['SQ']:
            outputs.append(self.sub_forward(encodings['SQ'], self.embed_SQ, self.attn_SQ, self.cnn_SQ))
        if NN_params['DVMA']:
            outputs.append(self.sub_forward(encodings['DVMA'], self.embed_DVMA, self.attn_DVMA, self.cnn_DVMA))
        if NN_params['DVSA']:
            outputs.append(self.sub_forward(encodings['DVSA'], self.embed_DVSA, self.attn_DVSA, self.cnn_DVSA))
        if NN_params['DVEX']:
            outputs.append(self.sub_forward(encodings['DVEX'], self.embed_DVEX, self.attn_DVEX, self.cnn_DVEX))
        if NN_params['DVL2']:
            outputs.append(self.sub_forward(encodings['DVL2'], self.embed_DVL2, self.attn_DVL2, self.cnn_DVL2))
        x = torch.cat(outputs, 1)
        # x = torch.cat((x, feats), 1)
        x = self.drop(x)
        x = self.dense1(F.relu(x))
        # x = self.drop(x)
        # x = self.dense2(F.relu(x))
        # x = self.drop(x)
        # x = self.dense3(F.relu(x))
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

    def make_layers(self, vocab_len):
        embed = nn.Embedding(vocab_len, self.emb_len)
        attn = nn.TransformerEncoderLayer(d_model=self.emb_len, nhead=self.n_heads)  # attention through an encoder
        attn = nn.TransformerEncoder(attn, num_layers=self.num_layers)  # for stacked encoders
        cnn = nn.Conv1d(in_channels=self.emb_len, out_channels=self.cnn_size, kernel_size=3)
        return embed, attn, cnn

    def sub_forward(self, encoding, embed_layer, attn_layer, conv_layer):
        x = embed_layer(encoding.to(self.device))  # (N, L, E)
        x = x.transpose(0, 1).contiguous()  # (L, N, E)
        x = attn_layer(x)
        x = x.transpose(0, 1).contiguous()  # (N, L, E)
        x = x.transpose(1, 2).contiguous()  # (N , E, L)
        x = conv_layer(x)  # (N, Cout, L)
        x = F.relu(x)  # (N, Cout, L)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x

