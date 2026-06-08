import os
import torch

class Configs:
    def __init__(self):

        self.data_path = 'the path of your dataset/DAVIS/dataset.csv' 
        self.output_dir = 'the path of your dataset/output/CogNet_DTA_optimized_v1/'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        

        self.n_splits = 5             
        self.batch_size = 128           
        self.epochs = 1000             
        self.lr = 5e-4                
        self.weight_decay = 1e-4      
        self.random_state = 42 
        

        self.esm_embedding_path = 'the path of your dataset/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        self.contact_map_dir = 'the path of your dataset/DAVIS/protein_contact_maps_esm' 
        

        self.d_model = 256          
        self.nhead = 8
        self.dropout = 0.2            
        self.drug_fp_size = 1024       
        self.drug_node_dim = 78       
        self.protein_node_dim = 21
        

        self.mem_slots = 64
