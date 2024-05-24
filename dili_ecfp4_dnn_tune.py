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
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from fingerprint import smiles_to_ecfp4

### CONFIG ###
fp_size = 128
n_class = 2     # classification
tr_loader = None
val_loader = None
test_loader = None


### Torch Dataloader 함수 ###
def load_data(x_data, y_data, batch_size=32):
    x_data = torch.FloatTensor(x_data)
    y_data = torch.LongTensor(y_data)
    return DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=True)
   

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


### Dvice 설정 ###
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


### Training Function
def train_dnn(config):
    # 모델 정의
    model = DNN(fp_size, config['out_size'], n_class)
    model.to(device)

    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-10)
    loss_fn = torch.nn.CrossEntropyLoss()

    # train
    model.train()
    epochs = 10
    data_size = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0.0
        epoch_steps = 0

        for i, data in enumerate(tr_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct = (torch.argmax(outputs, dim=-1) == labels).sum().item()
            correct += running_correct
            epoch_steps += 1
            data_size += labels.size[0]
        
        epoch_loss = running_loss / epoch_steps
        epoch_accuracy = correct / data_size
        print(f"[{epoch+1} / {epochs}] loss: {epoch_loss:1.5f} accuracy: {epoch_accuracy:1.5f}")

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        model.eval()
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                val_step += 1
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    print("Training completed!!!")


def evaluate(model, test_loader):
    # test_loader = load_data(test_fps, test_labels, batch_size)
    test_loss = 0.0
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct/total


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "out_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_dnn),
            resources={"cpu":4, "gpu":gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space = config,
    )
    results = tuner.fit()
    best_trial = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.metrics["accuracy"]))

    best_trained_model = DNN(fp_size, best_trial.config['out_size'], n_class)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = evaluate(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))

def prepare_dataset(dataset, condition=None):
    x_name = "Canonical SMILES"
    y_name = "Label"
    df = pd.read_csv(dataset)
    X_data = [smiles_to_ecfp4(smile, fp_size) for smile in df[x_name]]
    y_data = df[y_name].to_list()

    if condition is not None:
        test_size= 0.2
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, shuffle=True, stratify=y_data)
        train_loader = load_data(X_train, y_train)
        val_loader = load_data(X_val, y_val)
        return train_loader, val_loader
    else:
        test_loader = load_data(X_data, y_data)
        return test_loader


if __name__ == "__main__":
    dataset_dir = "Dataset"
    log_dir = "Logs"
    train_dataset_file = 'Train_whole_dataset.csv'
    test_dataset_file = "Test_whole_dataset.csv"
    log_file = 'dili_ecfp4_dnn.log'
    train_dataset = os.path.join(dataset_dir, train_dataset_file)
    test_dataset = os.path.join(dataset_dir, test_dataset_file)
    logs = os.path.join(log_dir, log_file)
    seed = int(np.random.rand() * (2**32 - 1))

    # create dataloaders
    tr_loader, val_loader = prepare_dataset(train_dataset, "train")
    test_loader = prepare_dataset(test_dataset)

    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=100, gpus_per_trial=1)