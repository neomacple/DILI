import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs


### Load Data ####################################################################################
dataset = './processed_dataset' + '/DILIrank_Liew_Greene_Xu_cid_metabolism.csv'
df = pd.read_csv(dataset)
df_smiles = df['Canonical SMILES']
df_labels = df['Label']
df_cids = df['CID']
###################################################################################################

### MDS using Tanimoto Distance Matrix ###########################################################################
ecfp4 = []
fp_size = 1024
for smiles in df_smiles:
    m = Chem.MolFromSmiles(smiles)
    bit={}
    morganfp=AllChem.GetMorganFingerprintAsBitVect(m, useChirality=True, radius=2, nBits = fp_size, bitInfo=bit)
    ecfp4.append(morganfp)

n_ecfp4 = len(ecfp4)
dissim_matrix = np.ones((n_ecfp4, n_ecfp4))
for i in range(n_ecfp4):
    for j in range(n_ecfp4):
        similarity = DataStructs.TanimotoSimilarity(ecfp4[i], ecfp4[j])
        dissim_matrix[i,j] = 1.0 - similarity
fps = ['fp'+str(i+1) for i in range(fp_size)]

A = (-1/2) * dissim_matrix
one_vector = np.ones((n_ecfp4, 1))
H = np.identity(n_ecfp4) - np.matmul(one_vector, one_vector.T) / n_ecfp4
B= np.matmul(np.matmul(H, A), H)

eigen_values, eigen_vectors = np.linalg.eig(B)
pos_pos = np.where(eigen_values>0)[0]
eigen_values = eigen_values[pos_pos]
eigen_vectors = eigen_vectors[:, pos_pos]
sort_pos = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sort_pos]
eigen_vectors = eigen_vectors[:, sort_pos]

diag_vec = np.sqrt(np.diag(eigen_values))
X = np.matmul(eigen_vectors, diag_vec)
ecfp4 = np.array(ecfp4, dtype=np.float32)
Y = np.matmul(ecfp4.T, X)

fp_idx = 4
sns_df = pd.DataFrame(Y.T[:, :fp_idx], columns=fps[:fp_idx])
sns_df['Label'] = df["Voted_Label"]
sns.pairplot(sns_df, hue="Label")
plt.show()