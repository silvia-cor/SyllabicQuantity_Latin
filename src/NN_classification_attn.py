from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from tqdm import tqdm
import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from general.helpers import data_for_kfold
from dataset_prep.NN_dataloader import NN_DataLoader
from general.significance import significance_test
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)



class Penta_Attn(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, start_dim_dense):
        super().__init__()
        dim_dense = start_dim_dense
        self.emb_length = 32
        self.n_heads = 4
        self.cnn_size = 100
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
        self.maxpool = nn.MaxPool1d(3)
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
        #x = torch.cat((x, feats), 1)
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
        embed = nn.Embedding(vocab_len, self.emb_length)
        attn = nn.TransformerEncoderLayer(d_model=self.emb_length, nhead=self.n_heads)  # attention through an encoder
        attn = nn.TransformerEncoder(attn, num_layers=2) # for stacked encoders
        cnn = nn.Conv1d(in_channels=self.emb_length, out_channels=self.cnn_size, kernel_size=3)
        return embed, attn, cnn

    def sub_forward(self, encoding, embed_layer, attn_layer, conv_layer):
        x = embed_layer(encoding.to(device))  # (N, L, E)
        x = x.transpose(0, 1).contiguous()  # (L, N, E)
        x = attn_layer(x)
        x = x.transpose(0, 1).contiguous()  # (N, L, E)
        x = x.transpose(1, 2).contiguous()  # (N , E, L)
        x = conv_layer(x)  # (N, Cout, L)
        x = F.relu(x)  # (N, Cout, L)
        x = self.maxpool(x)
        L = x.size()[2]
        x = F.max_pool1d(x, L)  # (N, Cout, 1)
        x = x.squeeze(2)  # (N, Cout)
        return x



def xavier_uniform(model: nn.Module):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)


def _train(model, NN_params, train_generator, val_generator, save_path, n_epochs, patience):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    optimizer = optim.Adam(params=model.parameters())
    xavier_uniform(model)

    criterion = nn.CrossEntropyLoss().to(device)
    val_f1scores = []
    epochs_no_improv = 0
    val_f1score, val_f1max, tr_f1score = 0, 0, 0
    for epoch in range(n_epochs):
        # training
        epoch_loss = []
        with tqdm(train_generator, unit="batch") as train:
            model.train()
            all_preds = []
            all_labels = []
            for encodings, targets in train: #, feats in train:
                optimizer.zero_grad()
                targets = targets.to(device)
                #feats = feats.to(device)
                #preds = model(NN_params, encodings, feats)
                preds = model(NN_params, encodings)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.detach().clone().cpu().numpy())
                all_labels.extend(targets.detach().clone().cpu().numpy())
                tr_f1score = f1_score(all_labels, all_preds, average='macro')
                train.set_description(f'Epoch {epoch+1} loss={np.mean(epoch_loss):.5f} tr-macro-F1={tr_f1score:.3f}')

        #evaluation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for encodings, targets in val_generator: #, feats in val_generator:
                #feats = feats.to(device)
                #preds = model(NN_params, encodings, feats)
                preds = model(NN_params, encodings)
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.detach().clone().cpu().numpy())
                all_labels.extend(targets.numpy())
            val_f1score = f1_score(all_labels, all_preds, average='macro')
            print(f'Val_F1-max: {val_f1max:.3f} Val_F1: {val_f1score:.3f}')
            # for the first 50 epochs, it simply trains
            # afterwards, if after patience there is no improvement, early stop happens
            if epoch == 49:
                epochs_no_improv = 0
            if val_f1score > val_f1max:
                val_f1max = val_f1score
                torch.save(model.state_dict(), save_path)
                epochs_no_improv = 0
            else:
                epochs_no_improv += 1
            val_f1scores.append(val_f1score)
            if epochs_no_improv == patience and epoch > 49:
                print("Early stopping!")
                break

    model.load_state_dict(torch.load(save_path))
    model.train()
    for encodings, targets in val_generator: #, feats in val_generator:
        optimizer.zero_grad()
        targets = targets.to(device)
        #feats = feats.to(device)
        #preds = model(NN_params, encodings, feats)
        preds = model(NN_params, encodings)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    return val_f1scores


def _test(model, NN_params, test_generator, load_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for encodings, targets in test_generator: #, feats in test_generator:
            #feats = feats.to(device)
            #preds = model(NN_params, encodings, feats)
            preds = model(NN_params, encodings)
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_labels.extend(targets.numpy())
    return all_preds, all_labels


def NN_classification(dataset, NN_params, dataset_name, n_sent, pickle_path):
    os.makedirs(f'../NN_methods/{n_sent}sent', exist_ok=True)
    assert not (NN_params['FAKE'] and NN_params['SQ']), 'Can have FAKE or SQ channel, but not both'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            df = pickle.load(handle)
    else:
        df = {}
    method_name = _create_method_name(NN_params)
    if method_name in df:
        print(f'NN experiment {method_name} already done!')
    else:
        print(f'----- NN EXPERIMENT {method_name} -----')
        authors, titles, data, data_cltk, authors_labels, titles_labels = data_for_kfold(dataset)
        dataset = NN_DataLoader(data, data_cltk, authors_labels, NN_params, batch_size=128)
        #model = Penta_NN(NN_params, dataset.vocab_lens, len(authors), 205).to(device)
        model = Penta_Attn(NN_params, dataset.vocab_lens, len(authors), start_dim_dense=0).to(device)
        val_f1s = _train(model, NN_params, dataset.train_generator, dataset.val_generator, f'../NN_methods/{dataset_name}/{method_name}.pt',
                            n_epochs=500, patience=15)
        y_preds, y_te = _test(model, NN_params, dataset.test_generator, f'../NN_methods/{dataset_name}/{method_name}.pt')
        if 'True' not in df:
            df['True'] = {}
            df['True']['labels'] = y_te
        df[method_name] = {}
        df[method_name]['preds'] = y_preds
        df[method_name]['val_f1s'] = val_f1s
        macro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='macro')
        micro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='micro')
        df[method_name]['macroF1'] = macro_f1
        df[method_name]['microF1'] = micro_f1
    print('----- F1 SCORE -----')
    print(f'Macro-F1: {df[method_name]["macroF1"]:.3f}')
    print(f'Micro-F1: {df[method_name]["microF1"] :.3f}')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(df, handle)

    # significance test if SQ are in the features with another method
    # significance test is against the same method without SQ
    if ' + SQ' in method_name or ' + FAKE' in method_name:
        baseline = method_name.split(' + ')[0]
        if baseline in df:
            significance_test(df['True']['labels'], df[baseline]['preds'], df[method_name]['preds'], baseline)
        else:
            print(f'No {baseline} saved, significance test cannot be performed :/')
    else:
        print('No significance test requested')


# generates the name of the method used to save the results
def _create_method_name(NN_params):
    methods = []
    dv_methods = ['DVMA', 'DVSA', 'DVEX', 'DVL2']
    for method in dv_methods:
        if NN_params[method]:
            methods.append(method)
    if len(methods) == 4:
        method_name = 'ALLDV'
    else:
        method_name = ' + '.join(methods)
    if NN_params['SQ']:
        if method_name != '':
            method_name += ' + '
        method_name += 'SQ'
    if NN_params['FAKE']:
        if method_name != '':
            method_name += ' + '
        method_name += 'FAKE'
    return method_name
