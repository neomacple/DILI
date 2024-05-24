import pandas as pd 
import pubchempy as pcp 
import os

input_file = os.path.join("Data", "DILIrank.csv")
output_file = os.path.join("Results", "DILIrank_MW_SMILES.csv")

df = pd.read_csv(input_file)
#print(df.head())
#out_condition = df['vDILIConcern']!='Ambiguous DILI-concern'
#df = df[out_condition]
compounds = df['Compound Name']
molecular_weights = []
smiles = []
for i in range(len(df['Compound Name'])):
    #label = df.loc[i, "vDILIConcern"]
    compound_name = df.loc[i, 'Compound Name']
    cids = pcp.get_cids(compound_name, 'name', 'substance', list_return='flat')
    try:
        c = pcp.Compound.from_cid(cids[0])
        c_mw = c.molecular_weight
        c_smiles = c.canonical_smiles
    except:
        print("exception came to happen")
        c_mw = "ND"
        c_smiles = "ND"
    print(compound_name, ":", c_mw, ",", c_smiles)
    molecular_weights.append(c_mw)
    smiles.append(c_smiles)

df["MW"] = molecular_weights
df['SMILES'] = smiles

if not os.path.exists("Results"):
    os.mkdir("Results")
df.to_csv(output_file, index=False)