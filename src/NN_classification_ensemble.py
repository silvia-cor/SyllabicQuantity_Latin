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
from functools import reduce
from sklearn.metrics import f1_score
from general.helpers import data_for_kfold
from dataset_prep.NN_dataloader import NN_DataLoader
from general.significance import significance_test
from pathlib import Path
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


class Penta_NN(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, start_dim_dense, kernel_sizes=[3,4,5,6,7]):
        super().__init__()
        emb_length = 32
        cnn_out_size = 100
        self.n_labels = n_labels
        if NN_params['FAKE']:
            self.embed_FAKE, self.cnn_FAKE1, self.cnn_FAKE2, self.cnn_FAKE3, self.dense_FAKE1, self.dense_FAKE2 = \
                self.make_layers(vocab_lens['FAKE'], emb_length, cnn_out_size, kernel_sizes)
        if NN_params['SQ']:
            self.embed_SQ, self.cnn_SQ1, self.cnn_SQ2, self.cnn_SQ3, self.dense_SQ1, self.dense_SQ2 = \
                self.make_layers(vocab_lens['SQ'], emb_length, cnn_out_size, kernel_sizes)
        if NN_params['DVMA']:
            self.embed_DVMA, self.cnn_DVMA1, self.cnn_DVMA2, self.cnn_DVMA3, self.dense_DVMA1, self.dense_DVMA2 = \
                self.make_layers(vocab_lens['DVMA'], emb_length, cnn_out_size, [3])
        if NN_params['DVSA']:
            self.embed_DVSA, self.cnn_DVSA1, self.cnn_DVSA2, self.cnn_DVSA3, self.dense_DVSA1, self.dense_DVSA2 = \
                self.make_layers(vocab_lens['DVSA'], emb_length, cnn_out_size, [3])
        if NN_params['DVEX']:
            self.embed_DVEX, self.cnn_DVEX1, self.cnn_DVEX2, self.cnn_DVEX3, self.dense_DVEX1, self.dense_DVEX2 = \
                self.make_layers(vocab_lens['DVEX'], emb_length, cnn_out_size, [3])
        if NN_params['DVL2']:
            self.embed_DVL2, self.cnn_DVL21, self.cnn_DVL22, self.cnn_DVL23, self.dense_DVL21, self.dense_DVL22 = \
                self.make_layers(vocab_lens['DVL2'], emb_length, cnn_out_size, [3])
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

    def make_layers(self, vocab_len, emb_len, cnn_out_size, kernel_sizes):
        embed = nn.Embedding(vocab_len, emb_len)
        if len(kernel_sizes) > 1:
            cnn1 = nn.ModuleList([nn.Conv1d(emb_len, cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
            cnn2 = nn.ModuleList([nn.Conv1d(cnn_out_size, cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
            cnn3 = nn.ModuleList([nn.Conv1d(cnn_out_size, cnn_out_size, kernel_size) for kernel_size in kernel_sizes])
        else:
            cnn1 = nn.Conv1d(emb_len, cnn_out_size, kernel_sizes[0])
            cnn2 = nn.Conv1d(cnn_out_size, cnn_out_size, kernel_sizes[0])
            cnn3 = nn.Conv1d(cnn_out_size, cnn_out_size, kernel_sizes[0])
        dense1 = nn.Linear(cnn_out_size * len(kernel_sizes), 516)
        dense2 = nn.Linear(516, self.n_labels)
        return embed, cnn1, cnn2, cnn3, dense1, dense2

    def sub_forward(self, encoding, embed_layer, conv_stack, dense_layer1, dense_layer2):
        x = embed_layer(encoding.to(device))
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
                loss = Variable(loss, requires_grad=True)
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


def NN_classification_ensemble(dataset, NN_params, dataset_name, n_sent, pickle_path):
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
        dataset = NN_DataLoader(data, data_cltk, authors_labels, NN_params, batch_size=64)
        #model = Penta_NN(NN_params, dataset.vocab_lens, len(authors), 205).to(device)
        model = Penta_NN(NN_params, dataset.vocab_lens, len(authors), start_dim_dense=0).to(device)
        print('Total paramaters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Device:', device)
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
