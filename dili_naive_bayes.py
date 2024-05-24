import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

from fingerprint import *
from cross_validation import cv_decision_tree
from cross_validation import dt_model
from load_dataset import load_whole_dili
from load_dataset import load_dili

### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
# x_col = "Canonical SMILES"
x_col = "Pubchem FP"
y_col = "Label"


### LOAD DATA ###
train_df, val_df, test_df = load_dili(x_col, y_col, cv=False)

### Fingerprints ###
# fps = np.array([smiles_to_maccs(smiles) for smiles in train_df[x_col]])
# fps = np.array([smiles_to_rdkit_fp(smiles) for smiles in train_df[x_col]])
# fps = np.array([smiles_to_ecfp4(smiles, 256) for smiles in train_df[x_col]])
fps = np.array([strings_to_fp(fp) for fp in train_df[x_col]])
labels = train_df[y_col]
print("X_data shape:", fps.shape)

### Naive Bayesian ###
bnb = BernoulliNB()
bnb.fit(fps, labels)
print("Score:", bnb.score(fps, labels))
pred = bnb.predict(fps)
print(confusion_matrix(labels, pred))

### BEST MODEL TEST
# min_sample_split = best_metrics['best_n_split']
# min_sample_leaf = best_metrics['best_n_leaf']
# model = dt_model(depth=None, min_split=min_sample_split, \
#                     min_leaf=min_sample_leaf, seed=seed)
# model.fit(X_train, y_train)
# preds = model.predict(X_test)
# print('Confusion matrix:')
# cm = confusion_matrix(y_test, preds)
# TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
# print(cm)
# outs = "\nConfusion Matrix:\n"
# outs += f"  - TN: {TN}\n"
# outs += f"  - FP: {FP}\n"
# outs += f"  - FN: {FN}\n"
# outs += f"  - TP: {TP}"
# print(outs)

# print()
# print(best_metrics)
# test_accuracy = (TN+TP)/(TN+TP+FN+FP)
# print(f"Test accuacy: {test_accuracy:1.5f}")
# print()