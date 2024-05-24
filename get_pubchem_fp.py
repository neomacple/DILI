import os
import pubchempy as pcp
import pandas as pd
from tqdm import tqdm

### CONFIG ###
dataset_dir = "datasets"
dataset_file = 'train_whole_dataset.csv'
out_file = 'train_whole_dataset_pubchem_fp.csv'
dataset = os.path.join(dataset_dir, dataset_file)
out = os.path.join(dataset_dir, out_file)

df = pd.read_csv(dataset)
cids = df["CID"]
fps = []
for cid in tqdm(cids):
    compound = pcp.Compound.from_cid(cid)
    fps.append(compound.cactvs_fingerprint)
out_df = df[["ID", "CID", "Canonical SMILES", "Label"]]
out_df.loc[:, 'Pubchem FP'] = fps
out_df.to_csv(out, index=False)