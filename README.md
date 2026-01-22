# CogNet-DTA: Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory Retrieval and Attraction-Repulsion Interaction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Graph Library](https://img.shields.io/badge/PyG-2.3-3C9943)](https://www.pyg.org/)

> **Official implementation of the paper: "Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory Retrieval and Attraction-Repulsion Interaction".**

**CogNet-DTA** is a cognitive-inspired deep learning framework designed for robust Drug-Target Affinity (DTA) prediction. It addresses the limitations of "black-box" models by introducing a **Chemical Graph Memory Network (CGMN)** to mimic expert "experience" and an **Attraction-Repulsion** mechanism to model biophysical interactions. Crucially, it incorporates **Uncertainty Quantization (UQ)** via Monte Carlo Dropout to assess predictive reliability.

---

## 🚀 Key Features

* **🧠 Chemical Graph Memory Network (CGMN):**
    Unlike traditional models that learn from scratch, CogNet-DTA utilizes a learnable global memory bank (default $64 \times 256$) to store and retrieve canonical binding patterns (e.g., hydrophobic pockets, hydrogen bonds), enabling reasoning by analogy.

* **🧲 Attraction-Repulsion Mechanism:**
    The model predicts affinity not as a single scalar, but as the equilibrium between attractive potentials (functional group matching) and repulsive forces (steric hindrance):
    $$Affinity = Head_{attr}(F_{seq}, F_{mem}) - Head_{repul}(F_{struct}, F_{mem})$$

* **🧬 Spatial-Aware Protein Representation:**
    Combines evolutionary semantics (ESM-2) with 3D structural constraints. A **Distance-Weighted Attention (DW-Attn)** mechanism injects spatial bias from contact maps into the sequence representation, prioritizing long-range residue interactions.

* **📊 Uncertainty Quantization (UQ):**
    Implements Monte Carlo (MC) Dropout sampling during inference. This provides a confidence score alongside the affinity prediction, allowing researchers to filter out high-risk, unreliable predictions ("hallucinations").

---

## 🏗️ Model Architecture

The framework processes multi-modal inputs through four specialized pathways:

1.  **Drug Encoder:**
    * **Sequence:** ECFP4 fingerprints processed via an MLP bottleneck.
    * **Structure:** Molecular graphs processed via **GATv2Conv** with Super-Nodes to capture global topology.
2.  **Protein Encoder:**
    * **Sequence:** ESM-2 embeddings refined by **Distance-Weighted Attention** using contact maps.
    * **Structure:** Deep 2D CNNs extract hierarchical spatial motifs from contact maps.
3.  **CogNet Layer:** Fuses features and queries the Memory Bank to retrieve "chemical common sense."
4.  **Prediction Head:** Dual-pathway (Attraction vs. Repulsion) output.

---

## 📂 Dataset Preparation

The model requires three specific data components: **CSV Labels**, **ESM Embeddings**, and **Contact Maps**.

### 1. Directory Structure
Please organize your data directory as referenced in `configss.py`:

```text
data/
├── dataset.csv                        # Main label file (SMILES, Sequence, ID, Label)
├── embeddings/
│   └── protein_esm_embeddings.pkl     # Pre-computed ESM embeddings (Dict format)
└── protein_contact_maps_esm/          # Directory containing individual .npy files
    ├── P12345.npy                     # Filename must match 'Target_ID' in CSV
    ├── Q9XYZ1.npy
    └── ...
```
### 2. Main Data File (`.csv`)
```
The CSV file must contain the following columns (as used in `utilss.py`):

| Column Name | Description | Example |
| --- | --- | --- |
| `Drug` | SMILES string of the compound | `CC1=C(C=C(C=C1)NC(=O)...` |
| `Target Sequence` | Amino acid sequence | `MVSWGRFICLVV...` |
| `Target_ID` | Unique Protein ID (links to `.npy` map) | `NP_005148.2` |
| `Label` | Binding affinity (, , or ) | `7.36` |
```
### 3. Auxiliary Data
```
* **ESM Embeddings (`.pkl`):** A Python dictionary where keys are protein sequences and values are 1280-dimensional vectors.
* **Contact Maps (`.npy`):** Binary or probability matrices () representing residue-residue contacts. The filename must strictly match the `Target_ID` in the CSV.
```
---

## 🛠️ Installation & Requirements

1. **Clone the repository:**
```bash
git clone [https://github.com/aliveadult/CogNet-DTA.git](https://github.com/aliveadult/CogNet-DTA.git)
cd CogNet-DTA

```


2. **Environment Setup:**
The code relies on `torch`, `torch_geometric`, and `rdkit`.
```bash
# Example using Conda
conda create -n cognet python=3.8
conda activate cognet

# Install PyTorch
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

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

Run the main script. The code automatically performs 5-fold cross-validation, saves the best models, and runs Uncertainty Quantization (UQ) sampling () for the final evaluation.

```bash
python mains.py

```

### 3. Output Interpretation

The training log provides detailed metrics per epoch. Note that `UQ` represents the Mean Uncertainty (Standard Deviation) of the predictions.

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

CogNet-DTA achieves state-of-the-art performance on benchmark datasets. Below is a comparison on the **Davis** dataset (), demonstrating superior accuracy and ranking capability.

| Model | CI  | MSE  |   | UQ (Uncertainty) |
| --- | --- | --- | --- | --- |
| DeepDTA | 0.878 | 0.261 | 0.631 | - |
| GraphDTA | 0.889 | 0.238 | 0.684 | - |
| GS-DTA | 0.897 | 0.225 | 0.688 | - |
| **CogNet-DTA** | **0.911** | **0.189** | **0.721** | **0.014** |

---

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{Hang2026CogNetDTA,
  title={Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory Retrieval and Attraction-Repulsion Interaction},
  author={Hang, Huaibin and Pang, Shunpeng and Feng, Junxiao and Pang, Weina and Ma, WenJian and Jiang, Mingjian and Zhou, Wei and Zhang, Yuanyuan},
  journal={Journal of Chemical Theory and Computation},
  year={2026},
  publisher={American Chemical Society}
}

```

---

## 📧 Contact

For any questions regarding the code or dataset, please contact the corresponding author:
**Mingjian Jiang** (School of Information and Control Engineering, Qingdao University of Technology)

Email: `jiangmingjian@qut.edu.cn`.

```

```
