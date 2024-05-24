import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from fingerprint import smiles_to_maccs
from load_dataset import load_dili
from cross_validation import cv_knn
from cross_validation import knn_model


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
fp_size = 256
x_col = "Canonical SMILES"
y_col = "Label"


### LOAD DATA ###
train_df, test_df = load_dili(x_col, y_col, cv=True)
X_train = np.array([smiles_to_maccs(smiles) for smiles in train_df[x_col]])
y_train = train_df[y_col]
X_test = np.array([smiles_to_maccs(smiles) for smiles in test_df[x_col]])
y_test = test_df[y_col]


### CROSS VALIDATION SEETING ###
n_neighbors = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
leaf_sizes = [10, 20, 30, 40 ,50]
scores = {}
best_metrics = {
    'best_val_acc': 0.0,
    'best_neigh': 0,
    'best_leaf': 0,
    'seed': seed
}

for n_neigh in n_neighbors:
    for n_leaf in leaf_sizes:
        scores = cv_knn(5, (X_train, y_train), n_neigh, n_leaf, seed)
        train_accs = []
        val_accs = []

        for i, score in scores.items():
            train_acc, val_acc = score
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        train_acc_mean = sum(train_accs)/len(scores)
        val_acc_mean = sum(val_accs)/len(scores)

        if val_acc_mean > best_metrics['best_val_acc']:
            best_metrics['best_val_acc'] = val_acc_mean
            best_metrics['best_neigh'] = n_neigh
            best_metrics['best_leaf'] = n_leaf
        print(f"neighbos: {n_neigh:2d}, leaf: {n_leaf:2d}, \
            train_accuracy: {train_acc_mean:>1.5f}, \
            test_accuracy: {val_acc_mean:>1.5f}")

### BEST MODEL TEST
n_neighbors = best_metrics['best_neigh']
leaf_size = best_metrics['best_leaf']
model = knn_model(
    n_neigh=n_neighbors,
    leaf_size = leaf_size,
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Confusion matrix:')
cm = confusion_matrix(y_test, preds)
TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
print(cm)
outs = "\nConfusion Matrix:\n"
outs += f"  - TN: {TN}\n"
outs += f"  - FP: {FP}\n"
outs += f"  - FN: {FN}\n"
outs += f"  - TP: {TP}"
print(outs)

print()
print(best_metrics)
test_accuracy = (TN+TP)/(TN+TP+FN+FP)
print(f"Test accuacy: {test_accuracy:>1.5f}")
print()