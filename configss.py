import os
import torch

class Configs:
    def __init__(self):
        # --- 路径 ---
        self.data_path = '/media/8t/hanghuaibin/SaeGraphDTII/data/DAVIS/dataset_filtered_with_contact.csv' 
        self.output_dir = 'output/CogNet_DTA_optimized_v1/' # 更新输出路径名
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # --- 训练配置 ---
        self.n_splits = 5             
        self.batch_size = 128           
        self.epochs = 1000             
        self.lr = 5e-4                
        self.weight_decay = 1e-4      
        self.random_state = 42 
        
        # --- 蛋白嵌入 ---
        self.esm_embedding_path = '/media/8t/hanghuaibin/SaeGraphDTII/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        self.contact_map_dir = '/media/8t/hanghuaibin/SaeGraphDTII/data/DAVIS/protein_contact_maps_esm' 
        
        # --- 模型参数 ---
        self.d_model = 256          
        self.nhead = 8
        self.dropout = 0.2            
        self.drug_fp_size = 1024       
        self.drug_node_dim = 78       
        self.protein_node_dim = 21
        
        # --- CGMN 参数 ---
        self.mem_slots = 64  # 存储 64 个典型的结合模式