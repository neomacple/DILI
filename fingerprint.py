from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def mol_to_maccs(mol):
    #fingerprint = MACCSkeys.GenMACCSKeys(mol).ToList()
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    return fingerprint.ToList()

def smiles_to_maccs(smiles):
    mol = smiles_to_mol(smiles)
    return mol_to_maccs(mol)

def smiles_to_rdkit_fp(smiles):
    mol = smiles_to_mol(smiles)
    fpgen = AllChem.GetRDKitFPGenerator()
    return fpgen.GetFingerprint(mol).ToList()

def smiles_to_ecfp4(smiles, size=1024):
    mol = smiles_to_mol(smiles)
    #fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=size)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)    # for old version of rdkit
    return fpgen.GetFingerprint(mol).ToList()

def strings_to_fp(strings):
    fp = [int(bit) for bit in strings]
    return fp