"""
CogNet-DTA: Data Preprocessing and Label Transformation Pipeline
================================================================
This script handles the parsing of raw Drug-Target interaction datasets
(Davis, KIBA, Metz, ToxCast, PDBbind), performs rigorous mathematical 
transformations on the affinity labels, and generates standardized 
features for the CogNet-DTA architecture.

Author: Huaibin Hang et al.
Project: CogNet-DTA
"""

import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

# Configure logging for reproducibility and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def derive_affinity_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Core function for mathematical derivation and transformation of affinity labels.
    Converts raw measurements into continuous logarithmic scales for standardized regression.
    """
    logging.info(f"--- Processing Label Transformations for {dataset_name} ---")
    
    if dataset_name.lower() == 'davis':
        # Davis: Raw Kd is in nM. Convert to pKd: -log10(Kd * 10^-9)
        # We add a small epsilon to prevent log(0) if any artifact 0 values exist
        epsilon = 1e-10
        df['affinity_label'] = -np.log10(df['raw_Kd_nM'] * 1e-9 + epsilon)
        logging.info("Davis: Converted raw Kd (nM) to continuous pKd scale.")

    elif dataset_name.lower() == 'metz':
        # Metz: Raw Ki is typically in Molar. Convert to pKi: -log10(Ki)
        df['affinity_label'] = -np.log10(df['raw_Ki'] + 1e-10)
        logging.info("Metz: Converted raw Ki to continuous pKi scale.")

    elif dataset_name.lower() == 'kiba':
        # KIBA: Already a mathematically amalgamated continuous score.
        df['affinity_label'] = df['raw_kiba_score']
        logging.info("KIBA: Kept pre-calculated continuous KIBA scores.")

    elif dataset_name.lower() == 'toxcast':
        # ToxCast: Extract AC50 (uM) for active hits and convert to pAC50
        # Formula: pAC50 = -log10(AC50 * 10^-6)
        # Filter out purely inactive samples without AC50 values first
        df = df[df['raw_AC50_uM'].notnull()].copy()
        df['affinity_label'] = -np.log10(df['raw_AC50_uM'] * 1e-6 + 1e-10)
        logging.info("ToxCast: Converted raw AC50 (uM) to continuous pAC50 scale.")

    elif dataset_name.lower() == 'pdbbind':
        # PDBbind: Natively provided as -log10(Kd/Ki) in the database
        df['affinity_label'] = df['raw_log_affinity']
        logging.info("PDBbind: Utilized native continuous -log10(Kd/Ki) binding scale.")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return df

def generate_ligand_features(smiles: str, radius=2, n_bits=1024):
    """
    Generates ECFP4 fingerprints and molecular graphs for the ligand.
    Matches the ECFP-based sequence encoder and GATv2 structural encoder requirements.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # 1. ECFP4 Fingerprint (Radius = 2) for sequence encoder
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    ecfp_tensor = torch.tensor(list(fp), dtype=torch.float32)
    
    # 2. Extract basic 2D graph features for GATv2 (Nodes and Adjacency)
    # Note: Full graph construction with edge features is handled in dataset class
    num_atoms = mol.GetNumAtoms()
    
    return ecfp_tensor, num_atoms

def process_dataset(csv_path: str, dataset_name: str, output_dir: str):
    """
    Main pipeline: Loads raw data, transforms labels, extracts features, and saves.
    """
    logging.info(f"Starting preprocessing pipeline for {dataset_name}...")
    
    # 1. Load Raw Data
    if not os.path.exists(csv_path):
        logging.warning(f"Raw data file not found at {csv_path}. Skipping.")
        return
        
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} raw interaction pairs.")
    
    # 2. Label Transformation (Addresses Reviewer 3's core concern)
    df = derive_affinity_labels(df, dataset_name)
    
    # 3. Ligand Feature Extraction
    logging.info("Extracting ligand ECFP4 and Graph features...")
    valid_indices = []
    ecfp_features = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        ecfp, num_atoms = generate_ligand_features(smiles)
        
        if ecfp is not None:
            valid_indices.append(idx)
            ecfp_features.append(ecfp)
            
    # Filter out invalid SMILES
    df_valid = df.loc[valid_indices].copy()
    ecfp_tensor_stack = torch.stack(ecfp_features)
    
    # 4. Save Processed Data
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{dataset_name}_processed_labels.csv")
    out_pt = os.path.join(output_dir, f"{dataset_name}_ecfp.pt")
    
    df_valid.to_csv(out_csv, index=False)
    torch.save(ecfp_tensor_stack, out_pt)
    
    logging.info(f"Successfully processed {len(df_valid)} valid pairs.")
    logging.info(f"Saved cleaned CSV to {out_csv} and Tensors to {out_pt}\n")

if __name__ == "__main__":
    # Define paths (Placeholders for local paths)
    DATA_DIR = "./raw_data"
    OUT_DIR = "./processed_data"
    
    # Example execution suite
    datasets_to_process = [
        ('Davis', f"{DATA_DIR}/davis_raw.csv"),
        ('KIBA', f"{DATA_DIR}/kiba_raw.csv"),
        ('ToxCast', f"{DATA_DIR}/toxcast_raw.csv")
    ]
    
    for name, path in datasets_to_process:
        # Create dummy file for demonstration purposes if running locally
        if not os.path.exists(path):
            os.makedirs(DATA_DIR, exist_ok=True)
            if name == 'Davis':
                pd.DataFrame({'SMILES': ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'], 'raw_Kd_nM': [0.1]}).to_csv(path, index=False)
            elif name == 'ToxCast':
                pd.DataFrame({'SMILES': ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'], 'raw_AC50_uM': [10.0]}).to_csv(path, index=False)
            elif name == 'KIBA':
                pd.DataFrame({'SMILES': ['CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'], 'raw_kiba_score': [12.1]}).to_csv(path, index=False)

        process_dataset(path, name, OUT_DIR)
        
    logging.info("Data preprocessing pipeline completed.")