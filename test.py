import os
import numpy as np
import pandas as pd
from load_dataset import preprocessing


dataset_dir = "datasets"
train_data_file = "train_whole_dataset.csv"
test_data_file = "test_whole_dataset.csv"
train_dataset_file = os.path.join(dataset_dir, train_data_file)
test_dataset_file = os.path.join(dataset_dir, test_data_file)

cols = ['Canonical SMILES', "Label"]
df_train = pd.read_csv(train_dataset_file)
df_train = preprocessing(df_train, cols)
df_test = pd.read_csv(test_dataset_file)
df_test = preprocessing(df_test, cols)

df = pd.concat([df_train, df_test], axis=0)
print(df.head())
print(len(df_train))
print(len(df_test))
print(len(df))

