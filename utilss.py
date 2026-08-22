import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os 
import pickle 
from rdkit import RDLogger 

try:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR) 
except: pass

def encode_one_hot(value, allowable_set):
    if value not in allowable_set:
        return [False] * len(allowable_set)
    return [value == s for s in allowable_set]

def encode_one_hot_unknown(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == s for s in allowable_set]

def atom_features(atom):

    symbol_set = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 
        'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 
        'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 
        'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]
    
    features = (
        encode_one_hot_unknown(atom.GetSymbol(), symbol_set) +
        encode_one_hot(atom.GetDegree(), list(range(11))) +
        encode_one_hot_unknown(atom.GetTotalNumHs(), list(range(11))) +
        encode_one_hot_unknown(atom.GetImplicitValence(), list(range(11))) +
        [atom.GetIsAromatic()]
    )

    features = np.array(features).astype(np.float32)
    return features / (np.sum(features) + 1e-6)

class HGDDTIDataset(Dataset):
    def __init__(self, df, esm_embeddings, config, contact_map_dir):
        self.df = df
        self.config = config
        self.esm_embeddings = esm_embeddings 
        self.contact_map_dir = contact_map_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_smiles = row['Drug']
        protein_seq = row['Target Sequence']
        protein_id = row['Target_ID']
        affinity = float(row['Label']) 
            
        if protein_seq not in self.esm_embeddings: return None
        p_emb = torch.tensor(self.esm_embeddings[protein_seq], dtype=torch.float)
        if p_emb.dim() == 1:
            p_emb = p_emb.unsqueeze(0)
        elif p_emb.dim() > 2:
            p_emb = p_emb.squeeze()
        if p_emb.dim() != 2 or p_emb.size(-1) != self.config.protein_esm_dim:
            return None

        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None: return None
        

        atom_f = [atom_features(a) for a in mol.GetAtoms()]
        if not atom_f: return None
        x_d = torch.tensor(np.array(atom_f), dtype=torch.float)

        edge_index = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edge_index.extend([(i, j), (j, i)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2,0), dtype=torch.long)
        

        x_s = torch.zeros(1, x_d.size(1), dtype=torch.float)
        x_all = torch.cat([x_d, x_s], dim=0)
        num_d = x_d.size(0)
        s_idx = num_d
        for i in range(num_d):
            edge_index = torch.cat([edge_index, torch.tensor([[i, s_idx], [s_idx, i]], dtype=torch.long).t()], dim=1)

        data = Data(x=x_all, edge_index=edge_index, y=torch.tensor([affinity], dtype=torch.float)) 
        data.num_drug_nodes = torch.tensor([num_d])
        data.num_protein_nodes = torch.tensor([0]) 
        data.num_super_nodes = torch.tensor([1]) 
        
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.config.drug_fp_size)
        d_token = torch.tensor([int(b) for b in mol_fp.ToBitString()], dtype=torch.float)

        c_path = os.path.join(self.contact_map_dir, f"{protein_id}.npy")
        try:
            c_map = torch.tensor(np.load(c_path), dtype=torch.float)
        except: return None

        return data, d_token, p_emb, affinity, c_map

def load_data(config):
    df = pd.read_csv(config.data_path)
    df['Label'] = df['Label'].astype(float)
    with open(config.esm_embedding_path, 'rb') as f:
        esm_embeddings = pickle.load(f)
    return df, esm_embeddings 

def get_k_fold_data(df, n_splits, random_state):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, test_idx in kf.split(df): 
        folds.append((df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)))
    return folds

def collate_fn_combined(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None, None
    g_batch = Batch.from_data_list([i[0] for i in batch])
    d_batch = torch.stack([i[1] for i in batch])
    p_batch = pad_sequence([i[2] for i in batch], batch_first=True, padding_value=0.0)
    a_batch = torch.tensor([i[3] for i in batch], dtype=torch.float).view(-1, 1)
    c_list = [i[4] for i in batch]
    return g_batch, d_batch, p_batch, a_batch, c_list
