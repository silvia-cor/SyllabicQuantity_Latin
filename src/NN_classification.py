import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from NN_dataset_loader import NN_DatasetBuilder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Multichannel_CNN(nn.Module):
    def __init__(self, vocab_sizes, n_labels):
        super().__init__()
        self.embed1 = nn.Embedding(vocab_sizes[0], 5)
        self.embed2 = nn.Embedding(vocab_sizes[1], 100)
        self.embed3 = nn.Embedding(vocab_sizes[2], 32)
        #controlla dimensioni cnn
        self.cnn1 = nn.Conv1d(5, 50, 5)
        self.cnn2 = nn.Conv1d(100, 50, 5)
        self.cnn3 = nn.Conv1d(32, 50, 5)
        self.drop = nn.Dropout(0.5)
        self.flat = nn.Flatten()
        self.final = nn.Linear(150, n_labels)

    def forward(self, input1, input2, input3):
        #channel 1
        hid1 = self.embed1(input1)
        hid1 = hid1.transpose(1, 2).contiguous()
        hid1 = self.conv_block(hid1, self.cnn1)
        out1 = self.flat(hid1)
        # channel 2
        hid2 = self.embed2(input2).transpose(1, 2).contiguous()
        hid2 = self.conv_block(hid2, self.cnn2)
        out2 = self.flat(hid2)
        # channel 3
        hid3 = self.embed3(input3).transpose(1, 2).contiguous()
        hid3 = self.conv_block(hid3, self.cnn3)
        out3 = self.flat(hid3)

        merge_out = torch.cat((out1, out2, out3), 1)
        dropped = self.drop(merge_out)
        final_out = self.final(dropped)
        return final_out

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activated = F.relu(conv_out)  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activated, conv_out.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out


def _train(model, training_generator, n_epochs):
    model.train()
    adam = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(n_epochs):
        with tqdm(training_generator, unit="batch") as tepoch:
            losses = []
            for enc1, enc2, enc3, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                enc1 = enc1.to(device)
                enc2 = enc2.to(device)
                enc3 = enc3.to(device)
                targets = targets.to(device)
                adam.zero_grad()
                predictions = model(enc1, enc2, enc3)
                loss = criterion(predictions, targets)
                loss.backward()
                adam.step()
                losses.append(loss.item())
                tepoch.set_postfix(losses = np.mean(losses))

def _test(model, x_te):
    model.eval()
    with torch.no_grad():
        return model(x_te)





def classification(dataset):
    training_generator = dataset.training_generator
    vocab_lens = dataset.vocab_lens
    model = Multichannel_CNN(vocab_lens, len(dataset.authors))
    model.to(device)
    print(model)
    _train(model, training_generator, 5)


#preds = _test(model, x_te)


# list of authors that will be added in the dataset
if __name__ == '__main__':
    authors = ['Vitruvius', 'Cicero', 'Iulius_Caesar', 'Suetonius', 'Titus_Livius', 'Servius',
           'Ammianus_Marcellinus', 'Apuleius', 'Augustinus_Hipponensis', 'Aulus_Gellius',
           'Columella', 'Cornelius_Nepos', 'Curtius_Rufus', 'Quintilianus',
           'Sallustius', 'Seneca_maior', 'Cornelius_Tacitus', 'Minucius_Felix',
           'Plinius_minor', 'Hieronymus_Stridonensis', 'Beda']

    dataset_path = "../dataset"  # change here for directory location

    dataset = NN_DatasetBuilder(authors, dataset_path, n_sentences=10, batch_size=10)
    classification(dataset)




