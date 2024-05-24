<<<<<<< HEAD
import torch
from torch import nn
from torch.utils.data import Dataset
#from torchmetrics import Accuracy, AUROC, ConfusionMatrix

import numpy as np

# define device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


### DATASET CLASS ###
class DILI_DATASET(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data)
        self.y = torch.tensor(y_data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
    

### MODEL CLASS ###
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


### OPTIMIZE LEARNING RATE 
class LRScheduler():
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor # LR을 factor배로 감소시킴
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            #verbose=True
        )
 
    def __call__(self, loss):
        self.lr_scheduler.step(loss)


### EARLY STOPPING 
class EarlyStopping():
    def __init__(self, patience=10, verbose=True, delta=0.0, path="model_state_dict.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_loss == np.Inf:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Metric Functions
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.special import softmax

def compute_metrics(targets, predictions):
    
    # accuracy
    pred_labels = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(targets, pred_labels)

    # roc_auc score
    probabilities = softmax(predictions)
    roc_auc = roc_auc_score(targets, probabilities[:,1])

    # confusion matrix
    cm = confusion_matrix(targets, pred_labels).ravel()

    return accuracy, roc_auc, cm

# Training 함수
def train(model, tr_loader, test_loader, loss_fn, optimizer, epochs):
    # hyperparamters for training
    patience = 20
    lr_patience = 10
    model_path = "./models/dili_dnn_model_state_dict.pt"
    model.train()
    patience = patience
    early_stopping = EarlyStopping(patience=patience, delta=0.0, path=model_path)
    lr_scheduler = LRScheduler(optimizer=optimizer, patience=lr_patience, min_lr=1e-6, factor=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0.0
        for batch, (X, y) in enumerate(tr_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (torch.argmax(outputs, dim=-1) == y).sum().item()
        epoch_loss /= (batch + 1)
        epoch_correct /= len(tr_loader.dataset)
        print(f"Epoch {epoch+1}: Loss: {epoch_loss:1.5f}, Accuracy: {epoch_correct:1.5f}")
        epoch_val_loss, _, _, _ = evaluate(model, test_loader, loss_fn, "Validation")
        lr_scheduler(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break


# Evaludation 함수
def evaluate(model, dataloader, loss_fn, state):
    model.eval()
    test_loss = 0.0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            test_loss += loss_fn(predictions, y).item()
            probabilities = torch.nn.functional.softmax(predictions, dim=-1)
            probabilities = probabilities.detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            outputs.append(probabilities)
            targets.append(labels)
    test_loss /= (batch+1)
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    acc, auc, cm = compute_metrics(targets, outputs)
    print(f"-- {state}: Loss: {test_loss:>2.5f}, Accuracy: {acc:>2.5f}, AUC: {auc:>2.5f}")
    return test_loss, acc, auc, cm

=======
import torch
from torch import nn
from torch.utils.data import Dataset
#from torchmetrics import Accuracy, AUROC, ConfusionMatrix

import numpy as np

# define device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


### DATASET CLASS ###
class DILI_DATASET(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data)
        self.y = torch.tensor(y_data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
    

### MODEL CLASS ###
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


### OPTIMIZE LEARNING RATE 
class LRScheduler():
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor # LR을 factor배로 감소시킴
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            #verbose=True
        )
 
    def __call__(self, loss):
        self.lr_scheduler.step(loss)


### EARLY STOPPING 
class EarlyStopping():
    def __init__(self, patience=10, verbose=True, delta=0.0, path="model_state_dict.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_loss == np.Inf:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Metric Functions
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.special import softmax

def compute_metrics(targets, predictions):
    
    # accuracy
    pred_labels = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(targets, pred_labels)

    # roc_auc score
    probabilities = softmax(predictions)
    roc_auc = roc_auc_score(targets, probabilities[:,1])

    # confusion matrix
    cm = confusion_matrix(targets, pred_labels).ravel()

    return accuracy, roc_auc, cm

# Training 함수
def train(model, tr_loader, test_loader, loss_fn, optimizer, epochs):
    # hyperparamters for training
    patience = 20
    lr_patience = 10
    model_path = "./models/dili_dnn_model_state_dict.pt"
    model.train()
    patience = patience
    early_stopping = EarlyStopping(patience=patience, delta=0.0, path=model_path)
    lr_scheduler = LRScheduler(optimizer=optimizer, patience=lr_patience, min_lr=1e-6, factor=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0.0
        for batch, (X, y) in enumerate(tr_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (torch.argmax(outputs, dim=-1) == y).sum().item()
        epoch_loss /= (batch + 1)
        epoch_correct /= len(tr_loader.dataset)
        print(f"Epoch {epoch+1}: Loss: {epoch_loss:1.5f}, Accuracy: {epoch_correct:1.5f}")
        epoch_val_loss, _, _, _ = evaluate(model, test_loader, loss_fn, "Validation")
        lr_scheduler(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break


# Evaludation 함수
def evaluate(model, dataloader, loss_fn, state):
    model.eval()
    test_loss = 0.0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            test_loss += loss_fn(predictions, y).item()
            probabilities = torch.nn.functional.softmax(predictions, dim=-1)
            probabilities = probabilities.detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            outputs.append(probabilities)
            targets.append(labels)
    test_loss /= (batch+1)
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    acc, auc, cm = compute_metrics(targets, outputs)
    print(f"-- {state}: Loss: {test_loss:>2.5f}, Accuracy: {acc:>2.5f}, AUC: {auc:>2.5f}")
    return test_loss, acc, auc, cm

>>>>>>> 772863a274eb6a0a836c8f6db049ba9b99d86e02
