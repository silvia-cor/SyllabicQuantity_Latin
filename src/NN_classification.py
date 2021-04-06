from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.autograd import Variable
from sklearn.metrics import f1_score
from general.helpers import data_for_kfold
from dataset_prep.NN_dataloader import NN_DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Penta_NN(nn.Module):
    def __init__(self, NN_params, vocab_lens, n_labels, kernel_sizes=[3,4,5]):
        super().__init__()
        dim_dense = 0
        if NN_params['FAKE']:
            self.embed_FAKE = nn.Embedding(vocab_lens['FAKE'], 16)
            self.cnn_FAKE = nn.ModuleList([nn.Conv1d(16, 100, kernel_size) for kernel_size in kernel_sizes])
            dim_dense += 100*(len(kernel_sizes))
        if NN_params['SQ']:
            self.embed_SQ = nn.Embedding(vocab_lens['SQ'], 16)
            self.cnn_SQ = nn.ModuleList([nn.Conv1d(16, 100, kernel_size) for kernel_size in kernel_sizes])
            dim_dense += 100 * (len(kernel_sizes))
        if NN_params['DVMA']:
            self.embed_DVMA= nn.Embedding(vocab_lens['DVMA'], 32)
            self.cnn_DVMA = nn.Conv1d(32, 100, 3)
            dim_dense += 100
        if NN_params['DVSA']:
            self.embed_DVSA= nn.Embedding(vocab_lens['DVSA'], 32)
            self.cnn_DVSA = nn.Conv1d(32, 100, 3)
            dim_dense += 100
        if NN_params['DVEX']:
            self.embed_DVEX= nn.Embedding(vocab_lens['DVEX'], 32)
            self.cnn_DVEX = nn.Conv1d(32, 100, 3)
            dim_dense += 100
        if NN_params['DVL2']:
            self.embed_DVL2= nn.Embedding(vocab_lens['DVL2'], 32)
            self.cnn_DVL2 = nn.Conv1d(32, 100, 3)
            dim_dense += 100
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(dim_dense, n_labels)

    def forward(self, NN_params, encodings):
        outputs = []
        if NN_params['FAKE']:
            hid_fake = self.embed_FAKE(encodings['FAKE'].to(device))
            hid_fake = [self.conv_block(hid_fake, conv) for conv in self.cnn_FAKE]
            hid_fake = torch.cat(hid_fake, 1)
            out_fake = self.flat(hid_fake)
            outputs.append(out_fake)
        if NN_params['SQ']:
            hid_sq = self.embed_SQ(encodings['SQ'].to(device))
            hid_sq = [self.conv_block(hid_sq, conv) for conv in self.cnn_SQ]
            hid_sq = torch.cat(hid_sq, 1)
            out_sq = self.flat(hid_sq)
            outputs.append(out_sq)
        if NN_params['DVMA']:
            hid_dvma = self.embed_DVMA(encodings['DVMA'].to(device))
            hid_dvma = self.conv_block(hid_dvma, self.cnn_DVMA)
            out_dvma = self.flat(hid_dvma)
            outputs.append(out_dvma)
        if NN_params['DVSA']:
            hid_dvsa = self.embed_DVSA(encodings['DVSA'].to(device))
            hid_dvsa = self.conv_block(hid_dvsa, self.cnn_DVSA)
            out_dvsa = self.flat(hid_dvsa)
            outputs.append(out_dvsa)
        if NN_params['DVEX']:
            hid_dvex = self.embed_DVEX(encodings['DVEX'].to(device))
            hid_dvex = self.conv_block(hid_dvex, self.cnn_DVEX)
            out_dvex = self.flat(hid_dvex)
            outputs.append(out_dvex)
        if NN_params['DVL2']:
            hid_dvl2 = self.embed_DVL2(encodings['DVL2'].to(device))
            hid_dvl2 = self.conv_block(hid_dvl2, self.cnn_DVL2)
            out_dvl2 = self.flat(hid_dvl2)
            outputs.append(out_dvl2)

        merge_out = reduce(lambda x,y: torch.cat((x,y)), outputs)
        drop_out = self.drop(merge_out)
        final_out = self.dense(F.relu(drop_out))
        return final_out

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input.transpose(1, 2).contiguous())
        activated =  F.relu(conv_out)
        max_out = F.max_pool1d(activated, conv_out.size()[2]).squeeze(2)
        return max_out

def _train(model, NN_params, train_generator, val_generator, save_path, n_epochs, epochs_stop = 10):
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    f1scores = []
    f1_max = 0
    epochs_no_improv = 0
    for epoch in range(n_epochs):
        #training
        epoch_loss = []
        with tqdm(train_generator, unit="batch") as train:
            model.train()
            for encodings, targets in train:
                train.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                targets = targets.to(device)
                preds = model(NN_params, encodings)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                train.set_postfix(loss=np.mean(epoch_loss))
        #evaluation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for encodings, targets in val_generator:
                preds = model(NN_params, encodings)
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.detach().clone().cpu().numpy())
                all_labels.extend(targets.numpy())
            f1score = f1_score(all_labels, all_preds, average='macro')
            if f1score > f1_max:
                f1_max = f1score
                epochs_no_improv = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improv += 1
            print('Val macro-F1:', f1score)
            f1scores.append(f1score)
            if epochs_no_improv == epochs_stop:
                print("Early stopping!")
                break
    model.train()
    for encodings, targets in val_generator:
        optimizer.zero_grad()
        targets = targets.to(device)
        preds = model(NN_params, encodings)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), save_path)
    return f1scores

def _test(model, NN_params, test_generator, load_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for encodings, targets in test_generator:
            preds = model(NN_params, encodings)
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_labels.extend(targets.numpy())
    return all_labels, all_preds


def NN_classification(dataset, NN_params, kfold, n_sent, pickle_path):
    assert isinstance(kfold, StratifiedKFold), 'Only kfold CV implemented for NN'
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
        authors, titles, data, authors_labels, titles_labels = data_for_kfold(dataset)
        for i, (trval_index, test_index) in enumerate(kfold.split(data, authors_labels)):
            print(f'----- K-FOLD EXPERIMENT {i + 1} -----')
            x_trval = data[trval_index]
            x_te = data[test_index]
            y_trval = authors_labels[trval_index]
            y_te = authors_labels[test_index]
            dataset = NN_DataLoader(x_trval, x_te, y_trval, y_te, NN_params, batch_size=128)
            model = Penta_NN(NN_params, dataset.vocab_lens, len(authors)).to(device)
            val_f1s = _train(model, NN_params, dataset.train_generator, dataset.val_generator, f'../NN_methods/{n_sent}sent/{method_name}.pt', n_epochs=50)
            y_all_te, y_all_pred = _test(model, NN_params, dataset.test_generator, f'../NN_methods/{n_sent}sent/{method_name}.pt')
            if 'True' not in df:
                df['True'] = {}
                df['True']['labels'] = y_all_te
            df[method_name] = {}
            df[method_name]['preds'] = y_all_pred
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


#generates the name of the method used to save the results
def _create_method_name(NN_params):
    methods = []
    dv_methods = ['DVMA', 'DVSA', 'DVEX', 'DVL2']
    method_name = ''
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