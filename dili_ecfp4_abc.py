import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from fingerprint import smiles_to_ecfp4
from cross_validation import cv_adaboost_model
from cross_validation import adaboost_model

from load_dataset import load_dili

def get_base_model(max_depth, seed):
    return DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=seed)


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
fp_size = 256
x_col = "Canonical SMILES"
y_col = "Label"


### LOAD DATA ###
train_df, test_df = load_dili(x_col, y_col, cv=True)
X_train = np.array([smiles_to_ecfp4(smiles) for smiles in train_df[x_col]])
y_train = train_df[y_col]
X_test = np.array([smiles_to_ecfp4(smiles) for smiles in test_df[x_col]])
y_test = test_df[y_col]


### CROSS VALIDATION SEETING ###
num_estimators = range(10, 60, 10)
learning_rates = [0.1, 0.5, 1, 5, 10]
depths = range(1,6)
scores = {}
best_metrics = {
    'best_val_acc': 0.0,
    'best_lr': 0.0,
    'best_depth': 0,
    'best_n_estimator': 0,
    'seed': seed
}

k = 5
for depth in depths:
    for lr in learning_rates:
        for n_est in num_estimators:
            # cv_adaboost_model(k, dataset, n_estimators, learning_rate, random_state, base_model):
            base_model = get_base_model(max_depth=depth, seed=seed)
            scores = cv_adaboost_model(k, (X_train, y_train), n_estimators=n_est, 
                                       learning_rate=lr, random_state=seed, 
                                       base_model=base_model)
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
                best_metrics['best_lr'] = lr
                best_metrics['best_depth'] = depth
                best_metrics['best_n_estimator'] = n_est
            print(f"depth: {depth}, learning_rate: {lr:>1.3f}, n_estimator: {n_est}, train_accuracy: {train_acc_mean:>1.4f}, val_accuracy: {val_acc_mean:>1.4f}")

### BEST MODEL TEST
best_depth = best_metrics['best_depth']
best_n_est = best_metrics['best_n_estimator']
best_lr = best_metrics['best_lr']

base_model = get_base_model(max_depth=best_depth, seed=seed)
model = adaboost_model(estimator=base_model, n_estimators=best_n_est, learning_rate=best_lr, random_state=seed)
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