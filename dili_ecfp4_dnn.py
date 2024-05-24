import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#from torchmetrics import Accuracy, AUROC, ConfusionMatrix
from sklearn.model_selection import train_test_split

from fingerprint import smiles_to_ecfp4
import torch_dnn


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
fp_size = 256
out_size = 256
batch_size = 32
lr = 0.001
epochs = 100
l2 = 1e-10



### LOAD DATA ###
def load_dataset(fp_size=1024):
    x_name = "Canonical SMILES"
    y_name = "Label"
    
    dataset_dir = "datasets"
    train_data_file = "train_whole_dataset.csv"
    test_data_file = "test_whole_dataset.csv"
    train_dataset_file = os.path.join(dataset_dir, train_data_file)
    test_dataset_file = os.path.join(dataset_dir, test_data_file)

    df_train = pd.read_csv(train_dataset_file)
    X_data = [smiles_to_ecfp4(smile, fp_size) for smile in df_train[x_name]]
    y_data = df_train[y_name].to_list()
    test_size= 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, shuffle=True, stratify=y_data)
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    df_test = pd.read_csv(test_dataset_file)
    X_data = [smiles_to_ecfp4(smile, fp_size) for smile in df_test[x_name]]
    y_data = df_test[y_name].to_list()
    X_test = torch.FloatTensor(X_data)
    y_test = torch.LongTensor(y_data)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    return train_dataset, val_dataset, test_dataset

### Torch DATALOADER ###
train_dataset, val_dataset, test_dataset = load_dataset(fp_size)
tr_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# model evaluation
device = torch_dnn.device
print("device:", device)
model = torch_dnn.DNN(fp_size, out_size, 2)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
<<<<<<< HEAD
model = torch_dnn.train(model, tr_loader, val_loader, loss_fn, optimizer, epochs)
=======
model = torch_dnn.train(model, tr_loader, val_loader, loss_fn, optimizer, epochs)
>>>>>>> 772863a274eb6a0a836c8f6db049ba9b99d86e02
