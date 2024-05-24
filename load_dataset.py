import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader,TensorDataset

from sklearn.model_selection import train_test_split

from fingerprint import smiles_to_ecfp4

# CONFIG
dataset_dir = "datasets"


def preprocessing(df, cols):
    # x_col = "Canonical SMILES"
    # y_col = "Label"
    # "Voted_Label" ---> "Label" 변경
    print(cols)
    col_to_replace = {'Voted_Label': "Label"}
    for old_key, new_key in col_to_replace.items():
        if old_key in df.columns:
            if new_key in df.columns:
                df.drop(columns=[new_key], inplace=True)
            df.rename(columns={old_key: new_key}, inplace=True)
    
    return df[cols]


# def load_whole_dili(x_col, y_col):
#     seed = int(np.random.rand() * (2**32 - 1))
#     test_size= 0.2
    
#     # load dataset
#     dataset_file = 'DILIrank_Liew_Greene_Xu_cid_metabolism.csv'
#     dataset = os.path.join(dataset_dir, dataset_file)
#     df = pd.read_csv(dataset)
#     df = preprocessing(df, [x_col, y_col])
#     n_notox, n_tox = df[y_col].value_counts().to_list()
    
#     # 500개 random index 생성
#     n_data = 500
#     np.random.seed(seed)
#     notox_idx_500 = np.random.randint(0, n_notox, n_data)
#     tox_idx_500 = np.random.randint(0, n_tox, n_data)
    
#     ### dataset을 dili 약물과 no-dili 약물로 구분지어 각각 500 개씩 뽑아서 train dataset으로,
#     ### 나머지는 validation dataset으로 이용
#     df = df.sample(frac=1).reset_index(drop=True)       # shuffle
#     df_notox = df[df['Label']==0].reset_index(drop=True)      # no-DILI 약물만 선택
#     df_notox_train = df_notox.loc[notox_idx_500, [x_col, y_col]]   # 500 개 random data
#     df_notox_train.reset_index(drop=True, inplace=True)
#     df_notox_test = df_notox.drop(df_notox_train.index).reset_index(drop=True)   # 나머지는 validation data로

#     df_tox = df[df['Label']==1].reset_index(drop=True)        # DILI 약물만 선택
#     df_tox_train = df_tox.loc[tox_idx_500, [x_col, y_col]]     # 500 개 random data
#     df_tox_train.reset_index(drop=True, inplace=True)
#     df_tox_test = df_tox.drop(df_tox_train.index).reset_index(drop=True)         # 나머지는 validation data로
    
#     ### TRAINING DATAFRAME & TESTING DATAFRAME ###
#     df_train = pd.concat([df_tox_train, df_notox_train], axis=0).sample(frac=1).reset_index(drop=True)
#     df_test = pd.concat([df_notox_test, df_tox_test], axis=0).sample(frac=1).reset_index(drop=True)

#     ### Training datasset --> training data & validation data
#     tr_x = df_train[x_col]
#     tr_y = df_train[y_col]
#     X_train, X_val, y_train, y_val = train_test_split(tr_x, tr_y, test_size=test_size, shuffle=True, stratify=tr_y)
    
#     tr_df = pd.DataFrame({x_col: X_train, y_col: y_train})
#     val_df = pd.DataFrame({x_col: X_val, y_col: y_val})
#     test_df = pd.DataFrame({x_col: df_test[x_col], y_col: df_test[y_col]})
#     tr_df.reset_index(inplace=True, drop=True)
#     val_df.reset_index(inplace=True, drop=True)
#     test_df.reset_index(inplace=True, drop=True)
    
#     return tr_df, val_df, test_df
    
    
    
def load_dili(x_col, y_col, cv=False):
    # x_col = "Canonical SMILES"
    # y_col = "Label"
    test_size= 0.2
    
    train_data_file = "train_whole_dataset_pubchem_fp.csv"
    test_data_file = "test_whole_dataset_pubchem_fp.csv"
    train_dataset_file = os.path.join(dataset_dir, train_data_file)
    test_dataset_file = os.path.join(dataset_dir, test_data_file)

    df_train = pd.read_csv(train_dataset_file)
    df_train = preprocessing(df_train, [x_col, y_col])
    df_test = pd.read_csv(test_dataset_file)
    df_test = preprocessing(df_test, [x_col, y_col])
    X_test = df_test[x_col]
    y_test = df_test[y_col]
    
    if cv:
        X_train = df_train[x_col]
        y_train = df_train[y_col]
        tr_df = pd.DataFrame({x_col: X_train, y_col: y_train})
        test_df = pd.DataFrame({x_col: X_test, y_col: y_test})
        tr_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)
        
        return tr_df, test_df
    else:
        X_train, X_val, y_train, y_val = train_test_split(df_train[x_col], df_train[y_col], 
                                                      test_size=test_size, shuffle=True, 
                                                      stratify=df_train[y_col])
    # X_train = torch.FloatTensor(X_train)
    # X_val = torch.FloatTensor(X_val)
    # y_train = torch.LongTensor(y_train)
    # y_val = torch.LongTensor(y_val)
    
    

    # X_test = torch.FloatTensor(X_data)
    # y_test = torch.LongTensor(y_data)
    
    # train_dataset = TensorDataset(X_train, y_train)
    # val_dataset = TensorDataset(X_val, y_val)
    # test_dataset = TensorDataset(X_test, y_test)
    
        tr_df = pd.DataFrame({x_col: X_train, y_col: y_train})
        val_df = pd.DataFrame({x_col: X_val, y_col: y_val})
        test_df = pd.DataFrame({x_col: X_test, y_col: y_test})
        tr_df.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)
    
        return tr_df, val_df, test_df



def load_cnn_dili(cols):
    # x_col = "Canonical SMILES"
    # y_col = "Label"
    test_size= 0.2
    train_data_file = "train_whole_dataset_pubchem_fp.csv"
    test_data_file = "test_whole_dataset_pubchem_fp.csv"
    train_dataset_file = os.path.join(dataset_dir, train_data_file)
    test_dataset_file = os.path.join(dataset_dir, test_data_file)

    df_train = pd.read_csv(train_dataset_file)
    df_train = preprocessing(df_train, cols)
    df_test = pd.read_csv(test_dataset_file)
    df_test = preprocessing(df_test, cols)
    
    y_col = "Label"
    tr_label = df_train[y_col]
    
    df_train, df_val, _, _ = train_test_split(df_train, tr_label, 
                                              test_size=test_size, 
                                              shuffle=True, 
                                              stratify=tr_label)
    
    return df_train, df_val, df_test



def load_whole_dili(cols):
    # x_col = "Canonical SMILES"
    # y_col = "Label"
    train_data_file = "train_whole_dataset_pubchem_fp.csv"
    test_data_file = "test_whole_dataset_pubchem_fp.csv"
    train_dataset_file = os.path.join(dataset_dir, train_data_file)
    test_dataset_file = os.path.join(dataset_dir, test_data_file)

    df_train = pd.read_csv(train_dataset_file)
    df_train = preprocessing(df_train, cols)
    df_test = pd.read_csv(test_dataset_file)
    df_test = preprocessing(df_test, cols)
    
    df = pd.concat([df_train, df_test], axis=0)
    
    return df