import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from fingerprint import *


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
dataset_dir = os.path.abspath("./datasets")
train_data_file = "train_whole_dataset.csv"
test_data_file = "test_whole_dataset.csv"


def load_dataset(data_file, train=False):
    x_name = "Canonical SMILES"
    y_name = "Label"
    
    #data_file = "train_whole_dataset.csv"
    data_path = os.path.join(dataset_dir, data_file)
    df = pd.read_csv(data_path)
    X_data = df[x_name].to_list()
    y_data = df[y_name].to_list()
    if train:
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, stratify=y_data)
        tr_maccs = torch.FloatTensor([smiles_to_maccs(smi) for smi in X_train])
        tr_rdkit = torch.FloatTensor([smiles_to_rdkit_fp(smi) for smi in X_train])
        tr_ecfp4 = torch.FloatTensor([smiles_to_ecfp4(smi) for smi in X_train])
        tr_labels = torch.LongTensor(y_train)
        val_maccs = torch.FloatTensor([smiles_to_maccs(smi) for smi in X_val])
        val_rdkit = torch.FloatTensor([smiles_to_rdkit_fp(smi) for smi in X_val])
        val_ecfp4 = torch.FloatTensor([smiles_to_ecfp4(smi) for smi in X_val])
        val_labels = torch.LongTensor(y_val)
        return ([tr_maccs, tr_rdkit, tr_ecfp4, tr_labels], [val_maccs, val_rdkit, val_ecfp4, val_labels])
    else:
        test_maccs = torch.FloatTensor([smiles_to_maccs(smi) for smi in X_data])
        test_rdkit = torch.FloatTensor([smiles_to_rdkit_fp(smi) for smi in X_data])
        test_ecfp4= torch.FloatTensor([smiles_to_ecfp4(smi) for smi in X_data])
        test_labels = torch.LongTensor(df[y_name].to_list())
        return([test_maccs, test_rdkit, test_ecfp4, test_labels])
    

train_data, val_data = load_dataset(train_data_file, train=True)
test_data = load_dataset(test_data_file)


### DNN Model 정의 ###
class 
class CNN_FP(nn.Module):
    def __init__(self, in_size, out_size, n_class):
        super().__init__()
        self.nn_maccs = nn.Linear(166, 255)
        self.nn_rdkit = nn.Linear(2048, 256)
        self.nn_ecfp4 = nn.Linear(1024, 255)
        self.conv2d_1 = nn.Conv2d(3, 128, 255, 3)
        self.pool_1 = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d()
        ,
        
        )

    def forward(self, inputs):
        fp1 = self.nn_maccs(inputs[0])
        fp2 = self.nn_rdkit(inputs[1])
        fp3 = self.nn_ecfp4(inputs[2])
        fp1 = fp1.view(25,25)
        fp2 = fp2.view(25,25)
        fp3 = fp3.view(25,25)
        fps = torch.stack(fp1, fp2, fp3)
        outs = 
        
        return logits