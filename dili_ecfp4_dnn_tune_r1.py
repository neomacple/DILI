import copy
import os
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset, random_split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ray import tune
from ray.train import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from fingerprint import smiles_to_ecfp4


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))


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
def load_dataset(dataset_dir=None):
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
def train_dnn(config, data_dir=None):    
    # model 정의
    fp_size = config['fp_size']
    out_size = config['out_size']
    n_class = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(fp_size, out_size, n_class)
    model.to(device)
    
    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-10)
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint = session.get_checkpoint()
    
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch=0
    
    # create dataloader
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_dir=data_dir)    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    
    
    epochs = 10
    for epoch in range(start_epoch, epochs):
        train(model, train_loader, loss_fn, optimizer, device)
        testing_loss, testing_acc = test(model, val_loader, loss_fn, device)
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        session.report(
            {"loss": testing_loss, "accuracy": testing_acc}, 
            checkpoint=checkpoint,
        )
    print("Training completed!!!")
    

def train(model, data_loader, loss_fn, optimizer, device):
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
    print(f"[Training] loss: {training_loss:1.5f} accuracy: {training_accuracy:1.5f}")
        

def test(model, data_loader, loss_fn, device):
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
    print(f"[Testing] loss: {testing_loss:1.5f} accuracy: {testing_accuracy:1.5f}")
    return testing_loss, testing_accuracy


max_num_epochs = 10
num_samples = 10
data_dir = os.path.abspath("./datasets")
load_dataset(data_dir)
config = {
    "out_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8, 16, 32, 64, 128]),
    "fp_size": tune.choice([128, 256])
    }
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2,
    )
# data_dir = "/home/macple/cheminfo/DILI/datasets"
gpus_per_trial = 1
result = tune.run(
    partial(train_dnn, data_dir=data_dir),
    resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    checkpoint_at_end=True)

# tuner = tune.Tuner(
#     train_dnn,
#     param_space=config,
# )
# results = tuner.fit()
# dfs = {result.path: result.metrics_dataframe for result in results}
# [d.mean_accuracy.plot() for d in dfs.values()]


# def evaluate(model, test_loader):
#     # test_loader = load_data(test_fps, test_labels, batch_size)
#     test_loss = 0.0
#     correct = 0
#     total = 0
    
#     model.eval()
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             outputs = model(inputs)
#             correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
#             total += labels.size(0)
#     return correct/total


# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#     config = {
#         "out_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([8, 16, 32, 64, 128])
#     }
#     scheduler = ASHAScheduler(
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
#     reporter = CLIReporter(
#         # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
#         metric_columns=["loss", "accuracy", "training_iteration"])
    
#     tuner = tune.Tuner(
#         tune.with_resources(
#             tune.with_parameters(train_dnn),
#             resources={"cpu":4, "gpu":gpus_per_trial}
#         ),
#         tune_config=tune.TuneConfig(
#             metric="loss",
#             mode="min",
#             scheduler=scheduler,
#             num_samples=num_samples,
#         ),
#         param_space = config,
#     )
#     results = tuner.fit()
#     best_trial = results.get_best_result("loss", "min")

#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.metrics["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.metrics["accuracy"]))

#     best_trained_model = DNN(fp_size, best_trial.config['out_size'], n_class)
#     best_trained_model.to(device)

#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(
#         best_checkpoint_dir, "checkpoint"))
#     best_trained_model.load_state_dict(model_state)

#     test_acc = evaluate(best_trained_model, device)
#     print("Best trial test set accuracy: {}".format(test_acc))




# if __name__ == "__main__":
#     dataset_dir = "Dataset"
#     log_dir = "Logs"
#     train_dataset_file = 'Train_whole_dataset.csv'
#     test_dataset_file = "Test_whole_dataset.csv"
#     log_file = 'dili_ecfp4_dnn.log'
#     train_dataset = os.path.join(dataset_dir, train_dataset_file)
#     test_dataset = os.path.join(dataset_dir, test_dataset_file)
#     logs = os.path.join(log_dir, log_file)
#     seed = int(np.random.rand() * (2**32 - 1))

#     # create dataloaders
#     tr_loader, val_loader = prepare_dataset(train_dataset, "train")
#     test_loader = prepare_dataset(test_dataset)

#     # You can change the number of GPUs per trial here:
#     main(num_samples=10, max_num_epochs=100, gpus_per_trial=1)