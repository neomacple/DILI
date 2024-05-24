import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from fingerprint import smiles_to_rdkit_fp
from cross_validation import cv_random_forest
from cross_validation import rf_model

from load_dataset import load_dili


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
fp_size = 256
x_col = "Canonical SMILES"
y_col = "Label"


### LOAD DATA ###
train_df, test_df = load_dili(x_col, y_col, cv=True)
X_train = np.array([smiles_to_rdkit_fp(smiles) for smiles in train_df[x_col]])
y_train = train_df[y_col]
X_test = np.array([smiles_to_rdkit_fp(smiles) for smiles in test_df[x_col]])
y_test = test_df[y_col]


## CROSS VALIDATION SEETING ###
sample_splits = range(2,10)
sample_leaf = range(1,10)
scores = {}
best_metrics = {
    'best_val_acc': 0.0,
    'best_n_split': 0,
    'best_n_leaf': 0,
    'seed': seed
}

k = 5
for n_split in sample_splits:
    for n_leaf in sample_leaf:
        scores = cv_random_forest(k, (X_train, y_train), n_split, n_leaf, seed)
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
            best_metrics['best_n_split'] = n_split
            best_metrics['best_n_leaf'] = n_leaf
        print(f"sample_split: {n_split:2d}, sample_leafs: {n_leaf:2d}, \
                train_accuracy: {train_acc_mean:>1.5f}, \
                val_accuracy: {val_acc_mean:>1.5f}")

### BEST MODEL TEST
min_sample_split = best_metrics['best_n_split']
min_sample_leaf = best_metrics['best_n_leaf']
model = rf_model(min_split=min_sample_split, min_leaf=min_sample_leaf, random_state=seed, max_depth=None)
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
print(f"Test accuacy: {test_accuracy:1.5f}")
print()