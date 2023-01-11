import os
import random
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import gc
import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# data prarameters
concat_nframes = 139            # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.95               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 7777                        # random seed
batch_size = 512                # batch size
num_epoch = 1000                   # the number of training epoch
early_stop_epoch = 5
learning_rate = 0.0001          # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved
log_path = './log.txt'

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 5               # the number of hidden layers
hidden_dim = 512                # the hidden dim


def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 2600000

    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)



class Classifier(nn.Module):
    def __init__(self, output_dim=41, hidden_size=512, num_layers=3):
        super(Classifier, self).__init__()

        self.rnn = nn.GRU(
            input_size=39,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(concat_nframes*hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.view(-1, concat_nframes, 39)
        x, h_n = self.rnn(x, None)
        x = self.fc(x)
        return x


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# l2 norm
def cal_regularization(model, weight_decay_l1=0.0, weight_decay_l2=0.0001):
    l1 = 0
    l2 = 0
    for i in model.parameters():
        l1 += torch.sum(abs(i))
        l2 += torch.sum(torch.pow(i, 2))
    return weight_decay_l1 * l1 + weight_decay_l2 * l2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # fix random seed
    same_seeds(seed)

    if args.mode == 'train':
        # preprocess data
        train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
        val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

        # get dataset
        train_set = LibriDataset(train_X, train_y)
        val_set = LibriDataset(val_X, val_y)

        # remove raw feature to save memory
        del train_X, train_y, val_X, val_y
        gc.collect()

        # get dataloader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16)


        # create model, define a loss function, and optimizer
        # model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
        model = Classifier().to(device)
        criterion = nn.CrossEntropyLoss() 
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        log = open(log_path, 'w')

        best_acc = 0.0
        early_stop = 0
        for epoch in range(num_epoch):
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            if epoch == 0:
                print('Optimizer AdamW')
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
            elif epoch == 10:
                print('Optimizer SGD')
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            
            # training
            model.train() # set the model to training mode
            for i, batch in enumerate(tqdm(train_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad() 
                outputs = model(features) 
                
                loss = criterion(outputs, labels)
                # loss.backward() 
                (loss + cal_regularization(model)).backward()
                optimizer.step() 
                
                _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                train_acc += (train_pred.detach() == labels.detach()).sum().item()
                train_loss += loss.item()
            
            # validation
            if len(val_set) > 0:
                model.eval() # set the model to evaluation mode
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(val_loader)):
                        features, labels = batch
                        features = features.to(device)
                        labels = labels.to(device)
                        outputs = model(features)
                        
                        loss = criterion(outputs, labels) 
                        
                        _, val_pred = torch.max(outputs, 1) 
                        val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                        val_loss += loss.item()

                    print('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                        epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                    ))
                    log.write('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}\n'.format(
                        epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                    ))

                    # if the model improves, save a checkpoint at this epoch
                    if val_acc > best_acc:
                        best_acc = val_acc
                        early_stop = 0
                        torch.save(model.state_dict(), model_path)
                        print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
                        log.write('saving model with acc {:.3f}\n'.format(best_acc/len(val_set)))
                    else:
                        early_stop += 1
                        print(f'not improved from {best_acc/len(val_set)}, early stop encounter: {early_stop}')
                        log.write(f'not improved from {best_acc/len(val_set)}, early stop encounter: {early_stop}\n')
                        if early_stop >= early_stop_epoch:
                            print('Early Stop')
                            log.write('Early stop\n')
                            break
            else:
                print('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
                ))
                log.write('[{:04d}/{:04d}] Train Acc: {:3.6f} Loss: {:3.6f}\n'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
                ))

        print(f'Best performance: {best_acc/len(val_set)}')
        log.write(f'Best performance: {best_acc/len(val_set)}\n')
        log.close()

        # if not validating, save the last epoch
        if len(val_set) == 0:
            torch.save(model.state_dict(), model_path)
            print('saving model at last epoch')

        del train_loader, val_loader, model
        gc.collect()

    if args.mode == 'test':
        # Test
        # load data
        test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
        test_set = LibriDataset(test_X, None)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)

        # load model
        # model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
        model = Classifier().to(device)
        model.load_state_dict(torch.load(model_path))

        test_acc = 0.0
        test_lengths = 0
        pred = np.array([], dtype=np.int32)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                features = batch
                features = features.to(device)

                outputs = model(features)

                _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)


        with open('prediction.csv', 'w') as f:
            f.write('Id,Class\n')
            for i, y in enumerate(pred):
                f.write('{},{}\n'.format(i, y))