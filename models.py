import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ChemicalMemoryBank(nn.Module):
    def __init__(self, mem_slots, d_model):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, d_model))
        self.query_proj = nn.Linear(d_model * 4, d_model)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x_query):
        q = self.query_proj(x_query) 
        scores = torch.matmul(q, self.memory.t())
        attn_weights = self.attend(scores)
        mem_out = torch.matmul(attn_weights, self.memory)
        return mem_out

class DistanceWeightedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.phi = nn.Sequential(
            nn.Linear(1, nhead),
            nn.LeakyReLU(0.2)
        )
        self.alpha = nn.Parameter(torch.ones(nhead))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, contact_map):
        N, D = x.shape
        d_k = D // self.nhead
        
        q = self.q_proj(x).view(N, self.nhead, d_k).transpose(0, 1)
        k = self.k_proj(x).view(N, self.nhead, d_k).transpose(0, 1)
        v = self.v_proj(x).view(N, self.nhead, d_k).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        bias = self.phi(contact_map.unsqueeze(-1)).permute(2, 0, 1)
        weighted_scores = scores + self.alpha.view(-1, 1, 1) * bias
        
        attn = self.dropout(F.softmax(weighted_scores, dim=-1))
        context = torch.matmul(attn, v)
        
        context = context.transpose(0, 1).contiguous().view(N, D)
        return self.out_proj(context)

class CogNetDTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.drug_seq_encoder = nn.Sequential(
            nn.Linear(config.drug_fp_size, d_model), 
            nn.ReLU(), 
            nn.Linear(d_model, d_model)
        )
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, d_model)
        
        self.dw_attn = DistanceWeightedAttention(d_model, config.nhead, config.dropout)
        
        self.contact_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.AdaptiveAvgPool2d((1, d_model)), 
            nn.Flatten(), 
            nn.Linear(16 * d_model, d_model)
        )
        self.structural_encoder = GATv2Conv(d_model, d_model)
        self.atom_proj = nn.Linear(78, d_model)

        self.memory_bank = ChemicalMemoryBank(mem_slots=64, d_model=d_model)

        self.attraction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), 
            nn.BatchNorm1d(d_model), 
            nn.LeakyReLU(0.2), 
            nn.Linear(d_model, 1)
        )
        self.repulsion_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.BatchNorm1d(d_model), 
            nn.LeakyReLU(0.2), 
            nn.Linear(d_model, 1)
        )

    def forward(self, graph_batch, drug_seq, protein_esm, contact_list):
        d_vec = self.drug_seq_encoder(drug_seq)
        
        p_vec_list = []
        for i in range(len(contact_list)):
            p_feat = F.relu(self.protein_esm_proj(protein_esm[i]))
            p_out = self.dw_attn(p_feat, contact_list[i])
            p_vec_list.append(p_out.mean(dim=0))
        p_vec = torch.stack(p_vec_list, 0)

        c_vec = self.contact_encoder(torch.stack(contact_list).unsqueeze(1))
        s_feat_all = self.structural_encoder(F.relu(self.atom_proj(graph_batch.x)), graph_batch.edge_index)
        
        s_vecs, start = [], 0
        for i in range(graph_batch.num_graphs):
            idx = start + graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item()
            s_vecs.append(s_feat_all[idx])
            start += (graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item() + graph_batch.num_super_nodes[i].item())
        s_vec = torch.stack(s_vecs)

        f_fused = torch.cat([d_vec, p_vec, s_vec, c_vec], dim=1) 
        m_vec = self.memory_bank(f_fused)

        attr_input = torch.cat([d_vec, p_vec, m_vec], dim=1)
        attr = self.attraction_head(attr_input)
        
        repu_input = torch.cat([s_vec, c_vec], dim=1)
        repu = self.repulsion_head(repu_input)

        return attr - repu
