import os
import pandas as pd 
import numpy as np 

input_file = os.path.join("Results", "DILIrank_MW_SMILES.csv")
raw_df = pd.read_csv(input_file)
pre_df = raw_df.replace("ND", np.nan)
pre_df.dropna(axis=0, ignore_index=True, inplace=True)

raw_labels = pre_df['vDILIConcern'].unique()
dili_category = [label for label in raw_labels if label != 'Ambiguous DILI-concern']
dili_labels = {idx:label for idx, label in enumerate(dili_category)}

most_dili = pre_df[pre_df['vDILIConcern']==dili_labels[0]]
no_dili = pre_df[pre_df['vDILIConcern']==dili_labels[1]]
less_dili = pre_df[pre_df['vDILIConcern']==dili_labels[2]]
print(" ")
print("Most-DILI compounds #:", len(most_dili))
print("Less-DILI compounds #:", len(less_dili))
print("No-DILI compounds #:", len(no_dili))
print(" ")

df = pre_df[pre_df['vDILIConcern']!='Ambiguous DILI-concern']
df = df.reset_index(drop=True)
invert_dili_label = {label:idx for idx, label in dili_labels.items()}
new_labels = [invert_dili_label[label] for label in df['vDILIConcern']]
new_labels = np.array(new_labels)
new_labels[new_labels==0] = 2
new_labels[new_labels==1] = 0
new_labels[new_labels==2] = 1
print("No-DILI compounds #:", len(new_labels[new_labels==0]))
print("DILI compounds #:", len(new_labels[new_labels==1]))
df['Label'] = new_labels

output_file = os.path.join("Results", "DILIrank_MW_SMILES_LABELS.csv")
if not os.path.exists("Results"):
    os.mkdir("Results")
df.to_csv(output_file, index=False)