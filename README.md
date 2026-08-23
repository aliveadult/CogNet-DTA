# CogNet-DTA: Decoding Molecular Binding Equilibria through Uncertainty-Aware Memory Retrieval


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Graph Library](https://img.shields.io/badge/PyG-2.3-3C9943)](https://www.pyg.org/)

> **Official implementation of the paper: "Decoding Molecular Binding Equilibria through Uncertainty-Aware Memory Retrieval".**

**CogNet-DTA** is a memory-augmented deep learning framework designed for robust Drug-Target Affinity (DTA) prediction. It addresses poor generalization and the limited reliability of conventional "black-box" models by introducing a **Chemical Graph Memory Network (CGMN)** to retrieve canonical binding motifs from historical data. It further combines **Contact-Weighted Attention (CW-Atten)** with a **Chemo-Geometric Routing Module (CGRM)** to organize multi-modal drug-target representations into positive and negative data-driven predictive pathways under a shared memory prior. Finally, it incorporates **Uncertainty Quantification (UQ)** via Monte Carlo Dropout to assess predictive reliability.

<p align="center">
  <img
    alt="CogNet-DTA framework"
    src="https://github.com/user-attachments/assets/e3c7d972-e8a7-4316-9089-92ebdf516f07"
    width="720"
  />
</p>


## 🚀 Key Features

* **🧠 Chemical Graph Memory Network (CGMN):**
    Unlike traditional models that treat each drug-target pair as an isolated event, CogNet-DTA utilizes a learnable global memory bank (default $64 \times 256$) to retrieve canonical binding motifs from historical data, enabling memory-augmented reasoning by analogy.

* **🧲 Chemo-Geometric Routing Module (CGRM):**
    The model organizes affinity prediction into positive and negative data-driven components through modality-specific routing under a shared memory prior:
    $$Affinity = A_{Positive} - A_{Negative}$$

* **🧬 Contact-Weighted Protein Representation:**
    Combines ESM-2 evolutionary embeddings with ESM-derived residue-residue contact probability matrices. A **Contact-Weighted Attention (CW-Atten)** mechanism injects contact-map-derived spatial bias into protein sequence representations, emphasizing residues with higher predicted contact probabilities.

* **📊 Uncertainty Quantification (UQ):**
    Implements Monte Carlo (MC) Dropout sampling during inference. This provides a predictive uncertainty score alongside the affinity prediction, allowing researchers to filter out low-confidence, high-risk predictions during virtual screening.

---

## 🏗️ Model Architecture

The framework processes multi-modal inputs through four specialized pathways:

1.  **Drug Encoder:**
    * **Sequence:** ECFP4 fingerprints processed via an MLP bottleneck.
    * **Structure:** Molecular graphs processed via **GATv2Conv** and global readout to capture topological connectivity.
2.  **Protein Encoder:**
    * **Sequence:** ESM-2 embeddings refined by **Contact-Weighted Attention (CW-Atten)** using continuous residue-residue contact probability matrices.
    * **Structure:** Deep 2D CNNs extract hierarchical spatial motifs from contact probability maps.
3.  **CGMN Layer:** Queries the learnable memory bank to retrieve memory-enhanced interaction features and shared bio-interaction patterns.
4.  **CGRM Prediction Head:** Dual-pathway positive/negative output with final affinity computed as $A_{Positive} - A_{Negative}$.

---

## 📂 Dataset Preparation

The model requires three specific data components: **CSV Labels**, **ESM Embeddings**, and **Residue-Residue Contact Probability Maps**.

### 1. Directory Structure
Please organize your data directory as referenced in `configss.py`:

```text
data/
├── dataset.csv                        # Main label file (SMILES, Sequence, ID, Label)
├── embeddings/
│   └── protein_esm_embeddings.pkl     # Pre-computed ESM embeddings (Dict format)
└── protein_contact_maps_esm/          # Directory containing individual contact-probability .npy files
    ├── P12345.npy                     # Filename must match 'Target_ID' in CSV
    ├── Q9XYZ1.npy
    └── ...
```
### 2. Main Data File (`.csv`)
The CSV file must contain the following columns (as used in `utilss.py`):

| Column Name | Description | Example |
| --- | --- | --- |
| `Drug` | SMILES string of the compound | `CC1=C(C=C(C=C1)NC(=O)...` |
| `Target Sequence` | Amino acid sequence | `MVSWGRFICLVV...` |
| `Target_ID` | Unique Protein ID (links to `.npy` map) | `NP_005148.2` |
| `Label` | Continuous affinity, bioactivity, or potency value (e.g., $pK_i$, $pK_d$, or $pAC_{50}$) | `7.36` |

### 3. Auxiliary Data

* **ESM Embeddings (`.pkl`):** A Python dictionary where keys are protein sequences and values are 1280-dimensional vectors.
* **Contact Probability Maps (`.npy`):** Continuous residue-residue contact likelihood matrices in $[0,1]$, generated from ESM-based structural prediction. The filename must strictly match the `Target_ID` in the CSV.

## 🌐 Data Access
You can access the processed dataset files (including CSV labels, PDB structures, and ESM embeddings) via the following link:
> **Google Drive Path**: [https://drive.google.com/drive/folders/1U_bl2IDNV-FqyBD4tMJKbDQzEBiLYbin?hl=zh_CN](https://drive.google.com/drive/u/1/folders/1MpV9fiJ7mmkxy_C5DVnU8mFHe2Td7ly1)

---

## 🛠️ Installation & Requirements

1. **Clone the repository:**
```bash
git clone https://github.com/aliveadult/CogNet-DTA.git
cd CogNet-DTA

```


2. **Environment Setup:**
The code relies on `torch`, `torch_geometric`, and `rdkit`.
```bash
# Example using Conda
conda create -n cognet python=3.8
conda activate cognet

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Graph Dependencies (PyG)
pip install torch_geometric

# Install Chem & Utility Libraries
pip install rdkit pandas numpy tqdm scikit-learn

```



---

## 🏃‍♂️ Usage

### 1. Configuration

Modify `configss.py` to set your file paths and hyperparameters:

```python
class Configs:
    def __init__(self):
        # Data Paths
        self.data_path = './data/DAVIS/dataset.csv'
        self.esm_embedding_path = './data/embeddings/DAVIS_protein_esm_embeddings.pkl'
        self.contact_map_dir = './data/DAVIS/protein_contact_maps_esm'
        
        # Training Params
        self.n_splits = 5        # K-Fold splits
        self.mem_slots = 64      # Size of Chemical Memory Bank
        self.batch_size = 128

```

### 2. Training & Evaluation

Run the main script. The code automatically performs 5-fold cross-validation, saves the best models, and runs Uncertainty Quantification (UQ) sampling (default $N=20$ stochastic forward passes) for the final evaluation.

```bash
python mains.py

```

### 3. Output Interpretation

The training log provides detailed metrics per epoch. Note that `UQ` represents predictive uncertainty estimated as the standard deviation of Monte Carlo Dropout predictions.

```text
>>> Fold 1 | CogNet-DTA Start
Epoch 001 | MSE: 0.8520 | Pearson: 0.8501 | CI: 0.8902 | RM2: 0.7201 | UQ: 0.072
...
===============================================================================================
       CogNet-DTA Final K-Fold Summary Report (with UQ)
===============================================================================================
Mean Squared Error                                      | 86.19 ± 00.28
Pearson Correlation Coefficient                         | 82.91 ± 00.72
Concordance Index                                       | 89.91 ± 00.65
Modified Squared Correlation Coefficient                | 72.72 ± 00.12
Mean Uncertainty (Standard Deviation)                   | 00.06 ± 00.23
===============================================================================================

```

---

## 📊 Performance

CogNet-DTA achieves highly competitive performance across benchmark datasets including **Davis**, **KIBA**, **Metz**, **ToxCast**, and **PDBbind**. Below is a comparison on the **Davis** dataset, demonstrating strong predictive accuracy and ranking capability.

| Model | CI  | MSE  | $r_m^2$ | UQ (Uncertainty) |
| --- | --- | --- | --- | --- |
| DeepDTA | 0.878 | 0.261 | 0.631 | - |
| GraphDTA | 0.889 | 0.238 | 0.684 | - |
| GS-DTA | 0.897 | 0.225 | 0.688 | - |
| **CogNet-DTA** | **0.911** | **0.189** | **0.721** | **0.0142** |
---

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{


}

```
