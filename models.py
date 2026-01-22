import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ChemicalMemoryBank(nn.Module):
    """
    创新组件：化学图记忆网络 (CGMN)。
    通过全局 Parameter 矩阵存储 DTA 任务中的通用结合模式。
    """
    def __init__(self, mem_slots, d_model):
        super().__init__()
        # 记忆矩阵：存储 mem_slots 个全局经验向量
        self.memory = nn.Parameter(torch.randn(mem_slots, d_model))
        # 查询映射：将药靶联合特征映射至记忆检索空间
        self.query_proj = nn.Linear(d_model * 2, d_model)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x_query):
        # x_query: [Batch, d_model * 2]
        q = self.query_proj(x_query) 
        # 计算当前样本与记忆池的相似度得分
        scores = torch.matmul(q, self.memory.t()) # [Batch, mem_slots]
        attn_weights = self.attend(scores)
        # 检索记忆：加权聚合经验向量
        mem_out = torch.matmul(attn_weights, self.memory) # [Batch, d_model]
        return mem_out

class DistanceWeightedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.v_linear = nn.Linear(d_model, d_model)
        self.dynamic_bias = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, contact_map):
        v = self.v_linear(x)
        spatial_bias = self.dynamic_bias(contact_map.mean().view(1))
        return v * (1 + spatial_bias)

class DrugSequenceEncoder(nn.Module):
    def __init__(self, fp_size, config):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(fp_size, config.d_model * 2), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.d_model * 2, config.d_model))
    def forward(self, x): return self.proj(x)

class ProteinContactEncoder(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Linear(32, d_model), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, c_list):
        vecs = [self.proj(self.adaptive_pool(F.relu(self.conv1(c.unsqueeze(0).unsqueeze(0)))).flatten()) for c in c_list]
        return torch.stack(vecs, 0)

class StructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, config.d_model // config.nhead, heads=config.nhead, dropout=config.dropout)
        self.conv2 = GATv2Conv(config.d_model, config.d_model // config.nhead, heads=config.nhead, dropout=config.dropout)
        self.conv3 = GATv2Conv(config.d_model, config.d_model // config.nhead, heads=config.nhead, dropout=config.dropout)
    def forward(self, x, edge_index):
        h1 = F.elu(self.conv1(x, edge_index))
        h2 = F.elu(self.conv2(h1, edge_index))
        h3 = F.elu(self.conv3(h2, edge_index))
        return x + h1 + h2 + h3 

class CogNetDTA(nn.Module):
    """
    CogNet-DTA: Cognitive Network with Attraction-Repulsion for Drug-Target Affinity.
    """
    def __init__(self, drug_fp_size, config): 
        super().__init__()
        self.atom_proj = nn.Linear(78, config.d_model)
        self.drug_seq_encoder = DrugSequenceEncoder(drug_fp_size, config)
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, config.d_model)
        self.structural_encoder = StructuralEncoder(config.d_model, config)
        self.contact_encoder = ProteinContactEncoder(config.d_model, config.dropout) 
        self.dw_attn = DistanceWeightedAttention(config.d_model, config.nhead, config.dropout)
        
        # --- CGMN 核心组件 ---
        self.memory_bank = ChemicalMemoryBank(mem_slots=config.mem_slots, d_model=config.d_model)
        
        self.ln_d, self.ln_p = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.ln_s, self.ln_c = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        
        # 预测头增强
        self.attraction_head = nn.Sequential(nn.Linear(config.d_model * 3, config.d_model), nn.BatchNorm1d(config.d_model), nn.LeakyReLU(0.2), nn.Linear(config.d_model, 1))
        self.repulsion_head = nn.Sequential(nn.Linear(config.d_model * 3, config.d_model), nn.BatchNorm1d(config.d_model), nn.LeakyReLU(0.2), nn.Linear(config.d_model, 1))

    def forward(self, graph_batch, drug_seq, protein_esm, contact_list):
        # 1. 基础编码
        d_vec = self.drug_seq_encoder(drug_seq)       
        p_vec = torch.stack([self.dw_attn(F.relu(self.protein_esm_proj(protein_esm))[i], contact_list[i]) for i in range(len(contact_list))], 0)
        c_vec, s_feat = self.contact_encoder(contact_list), self.structural_encoder(F.relu(self.atom_proj(graph_batch.x)), graph_batch.edge_index)
        
        # 2. 超节点特征提取
        s_vecs, start = [], 0
        for i in range(graph_batch.num_graphs):
            idx = start + graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item()
            s_vecs.append(s_feat[idx])
            start += (graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item() + graph_batch.num_super_nodes[i].item())
        s_vec = torch.stack(s_vecs, 0)
        
        # 3. 记忆检索 (CGMN Logic)
        mem_query = torch.cat([d_vec, p_vec], dim=1)
        mem_info = self.memory_bank(mem_query)
        
        # 4. 融合预测 (原有特征 + 记忆特征)
        attr = self.attraction_head(torch.cat([self.ln_d(d_vec), self.ln_p(p_vec), mem_info], dim=1))
        repu = self.repulsion_head(torch.cat([self.ln_s(s_vec), self.ln_c(c_vec), mem_info], dim=1))
        
        return attr - repu, attr, repu