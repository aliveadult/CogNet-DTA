import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ChemicalMemoryBank(nn.Module):
    def __init__(self, mem_slots, d_model):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, d_model))
        self.query_proj = nn.Linear(d_model * 2, d_model)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x_query):
        # x_query: [Batch, d_model * 2]
        q = self.query_proj(x_query) 
        scores = torch.matmul(q, self.memory.t()) # [Batch, mem_slots]
        attn_weights = self.attend(scores)
        mem_out = torch.matmul(attn_weights, self.memory) # [Batch, d_model]
        return mem_out

class ContactWeightedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, nhead)
        )
        self.dropout = nn.Dropout(dropout)

    def _align_contact_map(self, contact_map, seq_len):
        contact_map = contact_map.float()
        if contact_map.dim() > 2:
            contact_map = contact_map.squeeze()
        if contact_map.dim() != 2:
            contact_map = torch.eye(seq_len, device=contact_map.device, dtype=contact_map.dtype)

        contact_map = contact_map[:seq_len, :seq_len]
        pad_h = seq_len - contact_map.size(0)
        pad_w = seq_len - contact_map.size(1)
        if pad_h > 0 or pad_w > 0:
            contact_map = F.pad(contact_map, (0, max(pad_w, 0), 0, max(pad_h, 0)))
        return contact_map

    def forward(self, x, contact_list, protein_mask=None):
        # x: [Batch, Seq_len, d_model], contact_list: List[[Seq_len, Seq_len]]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, max_len, _ = x.size()
        if protein_mask is None:
            protein_mask = torch.ones(batch_size, max_len, device=x.device, dtype=torch.bool)

        pooled_outputs = []
        scale = self.head_dim ** 0.5

        for i in range(batch_size):
            valid_len = int(protein_mask[i].sum().item())
            if valid_len == 0:
                valid_len = max_len

            h = x[i, :valid_len]
            contact_map = self._align_contact_map(contact_list[i].to(x.device), valid_len)

            q = self.q_linear(h).view(valid_len, self.nhead, self.head_dim).transpose(0, 1)
            k = self.k_linear(h).view(valid_len, self.nhead, self.head_dim).transpose(0, 1)
            v = self.v_linear(h).view(valid_len, self.nhead, self.head_dim).transpose(0, 1)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            spatial_bias = self.bias_mlp(contact_map.unsqueeze(-1)).permute(2, 0, 1)
            attn_weights = torch.softmax(attn_scores + spatial_bias, dim=-1)
            attn_weights = self.dropout(attn_weights)

            refined = torch.matmul(attn_weights, v)
            refined = refined.transpose(0, 1).contiguous().view(valid_len, self.d_model)
            refined = self.out_linear(refined)
            pooled_outputs.append(refined.mean(dim=0))

        return torch.stack(pooled_outputs, dim=0)

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
        self.cw_attn = ContactWeightedAttention(config.d_model, config.nhead, config.dropout)
        
        self.memory_bank = ChemicalMemoryBank(mem_slots=config.mem_slots, d_model=config.d_model)
        
        self.ln_d, self.ln_p = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.ln_s, self.ln_c = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        
        self.attraction_head = nn.Sequential(nn.Linear(config.d_model * 3, config.d_model), nn.BatchNorm1d(config.d_model), nn.LeakyReLU(0.2), nn.Linear(config.d_model, 1))
        self.repulsion_head = nn.Sequential(nn.Linear(config.d_model * 3, config.d_model), nn.BatchNorm1d(config.d_model), nn.LeakyReLU(0.2), nn.Linear(config.d_model, 1))

    def forward(self, graph_batch, drug_seq, protein_esm, contact_list):
        d_vec = self.drug_seq_encoder(drug_seq)       
        if protein_esm.dim() == 2:
            protein_esm = protein_esm.unsqueeze(1)
        protein_mask = protein_esm.abs().sum(dim=-1) > 0
        p_refined = F.relu(self.protein_esm_proj(protein_esm))
        p_vec = self.cw_attn(p_refined, contact_list, protein_mask)
        c_vec, s_feat = self.contact_encoder(contact_list), self.structural_encoder(F.relu(self.atom_proj(graph_batch.x)), graph_batch.edge_index)
        
        s_vecs, start = [], 0
        for i in range(graph_batch.num_graphs):
            idx = start + graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item()
            s_vecs.append(s_feat[idx])
            start += (graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item() + graph_batch.num_super_nodes[i].item())
        s_vec = torch.stack(s_vecs, 0)
        
        mem_query = torch.cat([d_vec, p_vec], dim=1)
        mem_info = self.memory_bank(mem_query)
        
        attr = self.attraction_head(torch.cat([self.ln_d(d_vec), self.ln_p(p_vec), mem_info], dim=1))
        repu = self.repulsion_head(torch.cat([self.ln_s(s_vec), self.ln_c(c_vec), mem_info], dim=1))
        
        return attr - repu, attr, repu
