import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from dataset_prep.NN_dataloader import NN_DatasetBuilder
from general.visualization import val_performance_visual

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trichannel_LSTM(nn.Module):
    def __init__(self, vocab_lens, n_labels, kernel_sizes=[3,4,5]):
        super().__init__()
        self.embed1 = nn.Embedding(vocab_lens['syll'], 16)
        self.embed2 = nn.Embedding(vocab_lens['DVSA'], 100)
        self.embed3 = nn.Embedding(vocab_lens['DVEX'], 32)
        self.cnn1 = nn.ModuleList([nn.Conv1d(16, 100, kernel_size) for kernel_size in kernel_sizes])
        self.cnn2 = nn.Conv1d(100, 100, 5)
        self.cnn3 = nn.Conv1d(32, 100, 5)
        self.lstm1 = nn.LSTM(input_size=(100*len(kernel_sizes)), hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=256, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=256, batch_first=True)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_labels)

    def forward(self, enc_syll, enc_DVSA, enc_DVEX, batch_size):#, sent_lens):
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

class Bichannel_LSTM(nn.Module):
    def __init__(self, vocab_lens, n_labels):
        super().__init__()
        self.embed2 = nn.Embedding(vocab_lens['DVSA'], 100)
        self.embed3 = nn.Embedding(vocab_lens['DVEX'], 32)
        self.cnn2 = nn.Conv1d(100, 100, 5)
        self.cnn3 = nn.Conv1d(32, 100, 5)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=256, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=256, batch_first=True)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_labels)

    def forward(self, enc_DVSA, enc_DVEX, batch_size): #, sent_lens):
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

        merge_out = torch.cat((out2, out3), 1)
        drop_out = self.drop(merge_out)
        final_out = self.dense(F.relu(drop_out))
        return final_out

def init_hidden(batch_size):
    h_0 = Variable(torch.zeros(1, batch_size, 256)).to(device)
    c_0 = Variable(torch.zeros(1, batch_size, 256)).to(device)
    return (h_0, c_0)

def lstm_block(input, lstm_layer, batch_size):#, sent_lens):
    hidden = init_hidden(batch_size)
    #packed_input = pack_padded_sequence(input.to('cpu'), sent_lens, enforce_sorted=False, batch_first=True)
    #packed_out, _ = lstm_layer(packed_input.to(device), hidden)
    #unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)
    #sent_lens_idx = [length - 1 for length in sent_lens]
    #batch_idx = [i for i in range(batch_size)]
    #reshape_out = unpacked_out[batch_idx, sent_lens_idx, :]
    #input = input.transpose(1, 2).contiguous()
    output, hidden = lstm_layer(input, hidden)
    return output[:, -1, :]

def conv_block(input, conv_layer):
    conv_out = conv_layer(input.transpose(1, 2).contiguous())
    output = F.relu(conv_out)
    #max_out = F.max_pool1d(activated, conv_out.size()[2])#.squeeze(2)
    print(output.size())
    return output


def _train(model, train_generator, val_generator, save_path, n_epochs, epochs_stop = 10):
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
            for enc_syll, enc_DVSA, enc_DVEX, targets, sent_lens in train:
                train.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                enc_syll = enc_syll.to(device)
                enc_DVSA = enc_DVSA.to(device)
                enc_DVEX = enc_DVEX.to(device)
                targets = targets.to(device)
                preds = model(enc_syll, enc_DVSA, enc_DVEX, len(targets)) if isinstance(model, Trichannel_LSTM) else model(enc_DVSA, enc_DVEX, len(targets))
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
            for enc_syll, enc_DVSA, enc_DVEX, targets, sent_lens in val_generator:
                enc_syll = enc_syll.to(device)
                enc_DVSA = enc_DVSA.to(device)
                enc_DVEX = enc_DVEX.to(device)
                preds = model(enc_syll, enc_DVSA, enc_DVEX, len(targets)) if isinstance(model, Trichannel_LSTM) else model(enc_DVSA, enc_DVEX, len(targets))
                _, preds = torch.max(preds, dim=1)
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
    for enc_syll, enc_DVSA, enc_DVEX, targets, sent_lens in val_generator:
        optimizer.zero_grad()
        enc_syll = enc_syll.to(device)
        enc_DVSA = enc_DVSA.to(device)
        enc_DVEX = enc_DVEX.to(device)
        targets = targets.to(device)
        preds = model(enc_syll, enc_DVSA, enc_DVEX, len(targets)) if isinstance(model, Trichannel_LSTM) else model(enc_DVSA, enc_DVEX, len(targets))
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), save_path)
    return f1scores


def _test(model, test_generator, load_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    macrof1 = 0
    microf1 = 0
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for enc_syll, enc_DVSA, enc_DVEX, targets, sent_lens in test_generator:
            enc_syll = enc_syll.to(device)
            enc_DVSA = enc_DVSA.to(device)
            enc_DVEX = enc_DVEX.to(device)
            preds = model(enc_syll, enc_DVSA, enc_DVEX, len(targets)) if isinstance(model, Trichannel_LSTM) else model(enc_DVSA, enc_DVEX, len(targets))
            _, preds = torch.max(preds, dim=1)
            all_preds.extend(preds.detach().clone().cpu().numpy())
            all_labels.extend(targets.numpy())
        macrof1 = f1_score(all_labels, all_preds, average='macro')
        microf1 = f1_score(all_labels, all_preds, average='micro')
    print('Test macro-F1:', macrof1)
    print('Test micro-F1:', microf1)
    return macrof1, microf1


# list of authors that will be added in the dataset_prep

authors = ['Vitruvius', 'Cicero', 'Seneca', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius', 'Servius',
               'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
               'Columella', 'Florus', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus', 'Sallustius',
               'Seneca_maior', 'Sidonius_Apollinaris', 'Cornelius_Tacitus', 'Minucius_Felix',
               'Plinius_minor', 'Cornelius_Celsus', 'Beda', 'Hieronymus_Stridonensis']

dataset_path = "../dataset"  # change here for directory location

dataset = NN_DatasetBuilder(authors, dataset_path, n_sentences=10, batch_size=128)
model = Trichannel_LSTM(dataset.vocab_lens, len(dataset.authors)).to(device)
trichannel_val_loss = _train(model, dataset.train_generator, dataset.val_generator, '../pickles/Trichannel_LSTM.pt', n_epochs=50)
trichannel_test_loss = _test(model, dataset.test_generator, '../pickles/Trichannel_LSTM.pt')
model = Bichannel_LSTM(dataset.vocab_lens, len(dataset.authors)).to(device)
bichannel_val_loss = _train(model, dataset.train_generator, dataset.val_generator, '../pickles/Bichannel_LSTM.pt', n_epochs=50)
bichannel_test_loss = _test(model, dataset.test_generator, '../pickles/Bichannel_LSTM.pt')
models = {'Trichannel_LSTM':trichannel_val_loss, 'Bichannel_LSTM':bichannel_val_loss}
val_performance_visual(models, n_epochs=50)


