import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit import Chem


### Load Data ####################################################################################
dataset = './processed_dataset' + '/DILIrank_Liew_Greene_Xu_cid_metabolism.csv'
df = pd.read_csv(dataset)
features = ['MW', 'XLOGP', 'TPSA', 'H-Bond ACC', 'H-Bond Donor', 'Heavy_Atom_Count', 'Voted_Label']
df_data = df[features]
print(df_data.info())
###################################################################################################

### Data Processing ####################################################################################
# '.'으로 들어가 있는 데이터 제거
drop_index = []
for feature in features:
    idx = df_data[df_data[feature]=='.'].index.to_list()
    drop_index.extend(idx)
drop_index = list(set(drop_index))

df_data.drop(drop_index, inplace=True)
df_data.reset_index(drop=True, inplace=True)
df_data = df_data.astype('float')       # data type을 float로 전환

for feature in features[:-1]:       
    m = df_data[feature].mean()
    df_data[feature] = df_data[feature].fillna(m)       # 평균을 구해서 비어있는 값을 평균으로 매꿈
    df_data[feature] = (df_data[feature] - m)
    #df_data[feature] = (df_data[feature] - m) / df_data[feature].std()  # 데이터 표준화

print(df_data.info())

###################################################################################################
X = df_data[features[:-1]].to_numpy()
X = X.T
X0_ids = df_data[df_data['Voted_Label']==0.0].index
X1_ids = df_data[df_data['Voted_Label']==1.0].index
X0 = X[:, X0_ids]
X1 = X[:, X1_ids]
X0_mean = np.mean(X0, axis=1, keepdims=True)
X1_mean = np.mean(X1, axis=1, keepdims=True)
SB = np.matmul((X0_mean - X1_mean), (X0_mean - X1_mean).T)
SW = np.matmul((X0-X0_mean), (X0-X0_mean).T) + np.matmul((X1 - X1_mean), (X1 - X1_mean).T)
S = np.matmul(np.linalg.inv(SW), SB)
eigen_val, eigen_vec = np.linalg.eig(S)
print(eigen_val)
print()
print(eigen_vec)
print()
new_X = np.matmul(eigen_vec.T, X)
sns_df = pd.DataFrame(new_X.T, columns=features[:-1])
sns_df['Label'] = df_data["Voted_Label"]
sns.pairplot(sns_df, hue='Label')
plt.show()