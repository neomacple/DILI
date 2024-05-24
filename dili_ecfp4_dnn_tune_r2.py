import copy
import os
import tempfile
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset, random_split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

import matplotlib.pyplot as plt
from fingerprint import smiles_to_ecfp4


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
dataset_dir = os.path.abspath("./datasets")

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
 
### DNN Model 정의 ###
class DNN(nn.Module):
    def __init__(self, in_size, out_size, n_class):
        super().__init__()
        self.dnn_stack = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(out_size, n_class),
        )

    def forward(self, input):
        logits = self.dnn_stack(input)
        return logits


   
### Dataset 
def load_dataset(fp_size=1024):
    x_name = "Canonical SMILES"
    y_name = "Label"
    
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
    
    
### Training Function
def train_dnn(config):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # model 정의
    fp_size = config['fp_size']
    out_size = config['out_size']
    n_class = 2
    
    model = DNN(fp_size, out_size, n_class)
    model.to(device)
    
    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-10)
    loss_fn = torch.nn.CrossEntropyLoss()
        
    # create dataloader
    train_dataset, val_dataset, test_dataset = load_dataset(fp_size=fp_size)    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)  
    
    epochs = 500
    best_val_loss = np.Inf
    best_val_acc = np.Inf
    best_epoch = 0
   
    for epoch in range(epochs):
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = model_test(model, val_loader, loss_fn, device)
        test_loss, test_acc = model_test(model, test_loader, loss_fn, device)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, 'model.pth')
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            
            train.report({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "best_val_loss": best_val_loss, 
                "best_val_accuracy": best_val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "best_epoch": best_epoch}, 
                checkpoint=checkpoint)

    print("Training completed!!!")
    

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0.0
    epoch_steps = 0
    data_size = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
        epoch_steps += 1
        data_size += labels.size(0)
        
    training_loss = running_loss / epoch_steps
    training_accuracy = correct / data_size
    #print(f"[Training] loss: {training_loss:1.5f} accuracy: {training_accuracy:1.5f}")
    return training_loss, training_accuracy
        

def model_test(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    epoch_steps = 0
    data_size = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        running_loss += loss.item()
        correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
        epoch_steps += 1
        data_size += labels.size(0)
        
    testing_loss = running_loss / epoch_steps
    testing_accuracy = correct / data_size
    #print(f"[Testing] loss: {testing_loss:1.5f} accuracy: {testing_accuracy:1.5f}")
    return testing_loss, testing_accuracy


max_num_epochs = 100
num_samples = 50
load_dataset()
config = {
    "out_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #"lr": tune.loguniform(1e-4, 1e-1),
    "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "fp_size": tune.choice([128, 256, 512])
    }
scheduler = ASHAScheduler(
    metric="best_val_accuracy",
    mode="max",
    max_t=max_num_epochs,
    grace_period=20,
    reduction_factor=2,
    )

tuner = tune.Tuner(
    train_dnn,
    # tune_config=tune.TuneConfig(scheduler=scheduler),
    run_config = train.RunConfig(storage_path=os.path.abspath("./logs"), name="test_experiment"),
    tune_config=tune.TuneConfig(
        num_samples=50,
        scheduler=scheduler,
    ),
    param_space=config,
)
result_grid = tuner.fit()
# best_result = result_grid.get_best_result(metric="best_val_accuracy", mode="max")
# with best_result.checkpoint.as_directory() as checkpoint_dir:
#      state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pth'))
# model = DNN(in_size=best_result.config['fp_size'], out_size=best_result.config['out_size'], n_class=2)
# model.load_state_dict(state_dict)
# _, _, test_dataset = load_dataset(best_result.config['fp_size'])
# test_loader = DataLoader(test_dataset, batch_size=best_result.config['batch_size'], shuffle=True)
# loss_fn = torch.nn.CrossEntropyLoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loss, test_acc = model_test(model, test_loader, loss_fn, device)
# print(f"test loss: {test_loss:>1.5f}, test_acc: {test_acc:>1.5f}")


dfs = {result.path: result.metrics_dataframe for result in result_grid}

fig, axs = plt.subplots(2,2)
for df in dfs.values():
    # epochs = df.iloc[:,0]
    x_data = range(1, len(df)+1)
    for idx, col in enumerate(df.columns[0:4]):
        i = idx // 2
        j = idx % 2
        axs[i,j].plot(x_data, df[col])
        axs[i,j].set_title(col)
plt.show()

result_df = result_grid.get_dataframe()
result_df = result_df.sort_values(by='best_val_accuracy', ascending=False)
result_df.to_csv("./results/out.csv", index=False)
print(result_df.head(5))
