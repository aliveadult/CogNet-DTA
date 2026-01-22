CogNet-DTA: Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory RetrievalOfficial implementation of "Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory Retrieval and Attraction-Repulsion Interaction"1.CogNet-DTA is a cognitive-inspired deep learning framework designed to robustly predict drug-target affinity (DTA). It addresses key limitations in current models—specifically the lack of "memory" for historical binding patterns and the absence of confidence estimation—by introducing a Chemical Graph Memory Network (CGMN) and an Attraction-Repulsion mechanism2222.🚀 Key Features🧠 Chemical Graph Memory Network (CGMN): Utilizes a learnable memory bank ($64 \times 256$) to store and retrieve canonical binding patterns (e.g., hydrophobic pockets, hydrogen bonds), mimicking expert-driven reasoning3333.🧲 Attraction-Repulsion Mechanism: A dual-pathway prediction head that explicitly models affinity as the equilibrium between attractive potentials (functional group matching) and repulsive forces (steric hindrance)4444.🧬 Spatial-Aware Protein Representation: Integrates evolutionary semantics (ESM-2) with 3D structural constraints. It uses a Distance-Weighted Attention (DW-Attn) mechanism to prioritize residues that are spatially proximal in the folded structure5555.📊 Uncertainty Quantization (UQ): Implements Monte Carlo (MC) Dropout during inference to provide a confidence score alongside affinity predictions, enabling the filtering of "hallucinated" high-risk candidates6666.🏗️ Model ArchitectureThe framework consists of four main modules7:Drug Encoder:Sequence: ECFP4 fingerprints processed via MLP8.Structure: Molecular graphs processed via GATv2 with Super-Nodes to capture global topology9.Protein Encoder:Sequence: ESM-2 embeddings refined by Distance-Weighted Attention (DW-Attn) using contact maps10.Structure: 2D CNN processing of Contact Maps to extract hierarchical spatial motifs11.CogNet Layer (Memory): A query-projection mechanism that retrieves relevant binding experiences from the global memory bank12.Prediction Head: Calculates final affinity via:$$\text{Affinity} = \text{Attraction}(F_{seq}, F_{mem}) - \text{Repulsion}(F_{struct}, F_{mem})$$📂 Dataset PreparationTo replicate the results, data must be structured specifically to handle multi-modal inputs (Sequences, Graphs, and Contact Maps).1. Directory StructureEnsure your data directory is organized as follows:Plaintextdata/
├── DAVIS/
│   ├── dataset_filtered_with_contact.csv  # Main data file
│   └── protein_contact_maps_esm/          # Directory containing .npy files
│       ├── P12345.npy
│       ├── Q9XYZ1.npy
│       └── ...
└── embeddings/
    └── DAVIS_protein_esm_embeddings.pkl   # Pre-computed ESM embeddings
2. Main Data File Format (.csv)The CSV file must contain the following columns:ColumnDescriptionExampleDrugSMILES string of the compoundCC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)...Target SequenceAmino acid sequenceMVSWGRFICLVVVTMATLSLAR...Target_IDUnique identifier (matches contact map filename)NP_005148.2LabelBinding affinity value ($pK_d$, $pK_i$, or $pIC_{50}$)7.363. Auxiliary DataESM Embeddings (.pkl): A dictionary mapping Target Sequence strings to their corresponding ESM embedding vectors (Dimension: 1280)13.Contact Maps (.npy): Binary or probability matrices ($L \times L$) representing residue-residue contacts, generated via tools like ESMFold or AlphaFold. Saved as NumPy arrays14.🛠️ InstallationClone the repository:Bashgit clone https://github.com/aliveadult/CogNet-DTA.git
cd CogNet-DTA
Install dependencies:It is recommended to use a generic Conda environment.Bashconda create -n cognet python=3.8
conda activate cognet

# Install PyTorch (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric

# Install RDKit and other utilities
pip install rdkit pandas numpy tqdm scikit-learn scipy
🏃‍♂️ UsageConfigurationModify configss.py to point to your data paths and adjust hyperparameters:Pythonclass Configs:
    def __init__(self):
        # Paths
        self.data_path = './data/DAVIS/dataset_filtered_with_contact.csv'
        self.contact_map_dir = './data/DAVIS/protein_contact_maps_esm'
        self.esm_embedding_path = './data/embeddings/DAVIS_protein_esm_embeddings.pkl'
        
        # Hyperparameters
        self.batch_size = 128
        self.lr = 5e-4
        self.mem_slots = 64  # Size of the Memory Bank
TrainingRun the main script to start K-Fold Cross-Validation training. This script automatically handles training, validation, and Uncertainty Quantization (UQ) evaluation.Bashpython mains.py
Evaluation OutputThe script prints detailed metrics per epoch, including MSE, Pearson Correlation, CI (Concordance Index), $r_m^2$, and UQ (Mean Uncertainty).PlaintextEpoch 050 | MSE: 0.2105 | Pearson: 0.8650 | CI: 0.8840 | RM2: 0.6950 | UQ: 0.0210
...
CogNet-DTA Final K-Fold Summary Report (with UQ)
===============================================================================================
MSE   | 00.19 ± 00.01
CI    | 00.91 ± 00.00
RM2   | 00.72 ± 00.01
UQ    | 00.01 ± 00.00
📊 Performance ComparisonCogNet-DTA achieves state-of-the-art performance across multiple benchmarks. Below is the performance summary on the Davis dataset ($T=20$ sampling for UQ)15151515:ModelCI ↑MSE ↓rm2​ ↑Uncertainty (Uq)DeepDTA0.8780.2610.631-GraphDTA0.8890.2380.684-GS-DTA0.8970.2250.688-CogNet-DTA0.9110.1890.7210.014📜 CitationIf you find this code or paper useful for your research, please cite:代码段@article{Hang2026CogNetDTA,
  title={Uncertainty-Aware Drug-Target Affinity Prediction via Cognitive Memory Retrieval and Attraction-Repulsion Interaction},
  author={Hang, Huaibin and Pang, Shunpeng and Feng, Junxiao and Pang, Weina and Ma, WenJian and Jiang, Mingjian and Zhou, Wei and Zhang, Yuanyuan},
  journal={Journal of Chemical Theory and Computation},
  year={2026},
  publisher={American Chemical Society}
}
📧 ContactFor questions or inquiries, please contact:Mingjian Jiang (Corresponding Author) - jiangmingjian@qut.edu.cn 16This repository implements the methods described in the paper provided.
