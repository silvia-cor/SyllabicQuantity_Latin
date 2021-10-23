from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from tqdm import tqdm
import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score
from pathlib import Path
from general.helpers import data_for_kfold
from general.significance import significance_test
from dataset_prep.NN_dataloader import NN_DataLoader
from NN_models.NN_cnn_cat import Penta_Cat
from NN_models.NN_attn import Penta_Attn
from NN_models.NN_cnn_shallow_ensemble import Penta_ShallowEnsemble
from NN_models.NN_cnn_deep_ensemble import Penta_DeepEnsemble
from NN_models.NN_lstm import Penta_Lstm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def NN_classification(dataset, NN_params, model_name, dataset_name, n_sent, pickle_path, batch_size=64):
    os.makedirs(f'../NN_methods/{n_sent}sent', exist_ok=True)
    available_models = ['cnn_cat', 'attn', 'cnn_deep_ensemble', 'cnn_shallow_ensemble', 'lstm']
    assert not (NN_params['FAKE'] and NN_params['SQ']), 'Can have FAKE or SQ channel, but not both'
    assert model_name in available_models, f'Model not implemented, available models: {available_models}'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            df = pickle.load(handle)
    else:
        df = {}
    method_name = _create_method_name(NN_params)
    if method_name in df:
        print(f'NN experiment {method_name} for model {model_name} already done!')
    else:
        print(f'----- NN EXPERIMENT {method_name} for model {model_name} -----')
        authors, titles, data, data_cltk, authors_labels, titles_labels = data_for_kfold(dataset)
        dataset = NN_DataLoader(data, data_cltk, authors_labels, NN_params, batch_size=batch_size)
        # model = Penta_NN(NN_params, dataset.vocab_lens, len(authors), 205).to(device)
        if model_name == 'cnn_cat':
            model = Penta_Cat(NN_params, dataset.vocab_lens, len(authors), start_dim_dense=205, device=device).to(device)
        elif model_name == 'attn':
            model = Penta_Attn(NN_params, dataset.vocab_lens, len(authors), start_dim_dense=0, device=device).to(device)
        elif model_name == 'cnn_deep_ensemble':
            model = Penta_DeepEnsemble(NN_params, dataset.vocab_lens, len(authors), device=device).to(device)
        elif model_name == 'cnn_shallow_ensemble':
            model = Penta_ShallowEnsemble(NN_params, dataset.vocab_lens, len(authors), device=device).to(device)
        elif model_name == 'lstm':
            model = Penta_Lstm(NN_params, dataset.vocab_lens, len(authors), start_dim_dense=0, device=device).to(device)
        else:
            model = None
        print('Total paramaters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Device:', device)
        val_f1s = _train(model, NN_params, dataset.train_generator, dataset.val_generator, f'../NN_methods/{dataset_name}/{method_name}.pt',
                            n_epochs=5000, patience=100)
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
            print(f'COMPARISON WITH BASELINE {baseline}')
            delta_macro = (df[method_name]['macroF1'] - df[baseline]['macroF1']) / df[baseline]['macroF1'] * 100
            delta_micro = (df[method_name]['microF1'] - df[baseline]['microF1']) / df[baseline]['microF1'] * 100
            print(f'Macro-F1 Delta %: {delta_macro:.2f}')
            print(f'Micro-F1 Delta %: {delta_micro:.2f}')
            significance_test(df['True']['labels'], df[baseline]['preds'], df[method_name]['preds'], baseline)
        else:
            print(f'No {baseline} saved, significance test cannot be performed :/')
    else:
        print('No significance test requested')


# generates the name of the method used to save the results
def _create_method_name(NN_params):
    methods = []
    dv_methods = ['DVMA', 'DVSA', 'DVEX', 'DVL2']
    method_name = 'BaseFeatures'
    for method in dv_methods:
        if NN_params[method]:
            methods.append(method)
    if len(methods) == 4:
        method_name += '+ ALLDV'
    elif len(methods) > 0:
        if method_name != '':
            method_name += ' + '
        method_name += ' + '.join(methods)
    if NN_params['SQ']:
        method_name += ' + SQ'
    if NN_params['FAKE']:
        method_name += ' + FAKE'
    return method_name


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
            for encodings, targets, feats in train:
                optimizer.zero_grad()
                targets = targets.to(device)
                feats = feats.to(device)
                preds = model(NN_params, encodings, feats)
                # preds = model(NN_params, encodings)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.detach().clone().cpu().numpy())
                all_labels.extend(targets.detach().clone().cpu().numpy())
                tr_f1score = f1_score(all_labels, all_preds, average='macro')
                train.set_description(f'Epoch {epoch+1} loss={np.mean(epoch_loss):.5f} tr-macro-F1={tr_f1score:.3f}')

        # evaluation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for encodings, targets, feats in val_generator:
                feats = feats.to(device)
                preds = model(NN_params, encodings, feats)
                # preds = model(NN_params, encodings)
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.detach().clone().cpu().numpy())
                all_labels.extend(targets.numpy())
            val_f1score = f1_score(all_labels, all_preds, average='macro')
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
            print(f'Val_F1-max: {val_f1max:.3f} Val_F1: {val_f1score:.3f}')
            if epochs_no_improv == patience and epoch > 49:
                print("Early stopping!")
                break

    model.load_state_dict(torch.load(save_path))
    model.train()
    for encodings, targets, feats in val_generator:
        optimizer.zero_grad()
        targets = targets.to(device)
        feats = feats.to(device)
        preds = model(NN_params, encodings, feats)
        # preds = model(NN_params, encodings)
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
        for encodings, targets, feats in test_generator:
            feats = feats.to(device)
            preds = model(NN_params, encodings, feats)
            # preds = model(NN_params, encodings)
            preds = torch.argmax(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_labels.extend(targets.numpy())
    return all_preds, all_labels

