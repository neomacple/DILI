import torch
from torch import nn


import numpy as np

from fingerprint import *
from load_dataset import load_cnn_dili


### GPU 사용 ###
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


### CONFIG ###
seed = int(np.random.rand() * (2**32 - 1))
x_col = "Canonical SMILES"
y_col = "Label"
fp_col = "Pubchem FP"
ecfp4_size = 256
maccs_size = 167
pubchem_size = 881



### LOAD DATA ###
cols = [x_col, fp_col, y_col]
train_df, vla_df, test_df = load_cnn_dili([x_col, fp_col, y_col])

tr_ecfp4 = np.array([smiles_to_ecfp4(smiles, ecfp4_size) for smiles in train_df[x_col]])
tr_maccs = np.array([smiles_to_maccs(smiles) for smiles in train_df[x_col]])
tr_pubchem = np.array([strings_to_fp(fp) for fp in train_df[fp_col]])
tr_labels = np.array(train_df[y_col])

print(tr_ecfp4.shape, tr_maccs.shape, tr_pubchem.shape)


### DNN Model 정의 ###
fp_size = 24
class DNN(nn.Module):
    def __init__(self,  out_size, k_size, n_class):
        super().__init__()
        self.ecfp4_linear = nn.Linear(ecfp4_size, fp_size * fp_size)
        self.maccs_linear = nn.Linear(maccs_size, fp_size * fp_size)
        self.pubchem_linear = nn.Linear(pubchem_size, fp_size * fp_size)
        self.conv = nn.Conv2d(3, out_size, k_size, padding=1)
        self.pool = nn.MaxPool2d()
        self.linear = nn.Linear()
        

    def forward(self, ecfp4, maccs, pubchem):
        x_ecfp4 = self.ecfp4_linear(ecfp4)
        x_ecfp4 = x_ecfp4.reshape(-1, fp_size, fp_size)
        x_maccs = self.maccs_linear(maccs)
        x_maccs = x_maccs.reshape(-1, fp_size, fp_size)
        x_pubchem = self.pubchem_linear(pubchem)
        x_pubchem = x_pubchem.reshape(-1, fp_size, fp_size)
        x = torch.cat((x_ecfp4, x_maccs, x_pubchem), axis=-1)
        x = self.conv(x)
        
        
        
        
        logits = self.dnn_stack(input)
        return logits