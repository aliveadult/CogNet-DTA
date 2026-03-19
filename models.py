import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ChemicalMemoryBank(nn.Module):
    """
    创新组件：化学图记忆网络 (CGMN)。
    符合论文 Theory：使用全模态融合特征作为 Query。
    """
    def __init__(self, mem_slots, d_model):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, d_model))
        # 论文要求拼接四种特征，这里假设每种特征维度均为 d_model，总维度 d_model * 4
        self.query_proj = nn.Linear(d_model * 4, d_model)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x_query):
        # x_query: [Batch, d_model * 4]
        q = self.query_proj(x_query) 
        scores = torch.matmul(q, self.memory.t()) # [Batch, mem_slots]
        attn_weights = self.attend(scores)
        mem_out = torch.matmul(attn_weights, self.memory) # [Batch, d_model]
        return mem_out

class DistanceWeightedAttention(nn.Module):
    """
    核心修改：符合论文 Theory 的空间偏置注意力。
    将二维 Contact Map 转化为 N x N 的 Bias Matrix 注入 Attention Score。
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 映射函数 phi: 将原始接触图数值映射为空间偏置
        # 论文中通常使用小型卷积或多层 MLP 学习偏置量
        self.phi = nn.Sequential(
            nn.Linear(1, nhead), # 为每个 head 学习一个独立的偏置
            nn.LeakyReLU(0.2)
        )
        self.alpha = nn.Parameter(torch.ones(nhead)) # 论文中的可学习缩放因子
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, contact_map):
        """
        x: [N, d_model] - 蛋白序列特征
        contact_map: [N, N] - 二维接触图
        """
        N, D = x.shape
        d_k = D // self.nhead
        
        q = self.q_proj(x).view(N, self.nhead, d_k).transpose(0, 1) # [H, N, d_k]
        k = self.k_proj(x).view(N, self.nhead, d_k).transpose(0, 1) # [H, N, d_k]
        v = self.v_proj(x).view(N, self.nhead, d_k).transpose(0, 1) # [H, N, d_k]

        # 1. 计算原始语义注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5) # [H, N, N]

        # 2. 生成矩阵级空间偏置 (Theory 核心)
        # contact_map: [N, N] -> [N, N, 1] -> [N, N, H] -> [H, N, N]
        bias = self.phi(contact_map.unsqueeze(-1)).permute(2, 0, 1)
        
        # 3. 注入偏置 (Score + alpha * Bias)
        weighted_scores = scores + self.alpha.view(-1, 1, 1) * bias
        
        attn = self.dropout(F.softmax(weighted_scores, dim=-1))
        context = torch.matmul(attn, v) # [H, N, d_k]
        
        context = context.transpose(0, 1).contiguous().view(N, D)
        return self.out_proj(context)

class CogNetDTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # 1. 特征编码器
        self.drug_seq_encoder = nn.Sequential(nn.Linear(config.drug_fp_size, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, d_model)
        
        # 空间感知注意力
        self.dw_attn = DistanceWeightedAttention(d_model, config.nhead, config.dropout)
        
        # 结构编码器
        self.contact_encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.AdaptiveAvgPool2d((1, d_model)), nn.Flatten(), nn.Linear(16 * d_model, d_model))
        self.structural_encoder = GATv2Conv(d_model, d_model)
        self.atom_proj = nn.Linear(78, d_model)

        # 2. 认知组件
        self.memory_bank = ChemicalMemoryBank(mem_slots=64, d_model=d_model)

        # 3. 预测头
        # 引力通道：处理序列及经验特征 [d_vec, p_vec, m_vec] -> 3 * d_model
        self.attraction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), 
            nn.BatchNorm1d(d_model), 
            nn.LeakyReLU(0.2), 
            nn.Linear(d_model, 1)
        )
        # 斥力通道：处理结构特征 [s_vec, c_vec] -> 2 * d_model
        self.repulsion_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.BatchNorm1d(d_model), 
            nn.LeakyReLU(0.2), 
            nn.Linear(d_model, 1)
        )

    def forward(self, graph_batch, drug_seq, protein_esm, contact_list):
        # --- A. 特征提取 ---
        # 药物序列特征
        d_vec = self.drug_seq_encoder(drug_seq)
        
        # 蛋白序列特征 (带矩阵级空间偏置)
        p_vec_list = []
        for i in range(len(contact_list)):
            p_feat = F.relu(self.protein_esm_proj(protein_esm[i])) # [N, d_model]
            p_out = self.dw_attn(p_feat, contact_list[i])
            p_vec_list.append(p_out.mean(dim=0)) # 池化为全局蛋白特征
        p_vec = torch.stack(p_vec_list, 0)

        # 结构特征
        c_vec = self.contact_encoder(torch.stack(contact_list).unsqueeze(1))
        s_feat_all = self.structural_encoder(F.relu(self.atom_proj(graph_batch.x)), graph_batch.edge_index)
        
        # 提取药物超节点结构特征
        s_vecs, start = [], 0
        for i in range(graph_batch.num_graphs):
            idx = start + graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item()
            s_vecs.append(s_feat_all[idx])
            start += (graph_batch.num_drug_nodes[i].item() + graph_batch.num_protein_nodes[i].item() + graph_batch.num_super_nodes[i].item())
        s_vec = torch.stack(s_vecs)

        # --- B. 认知记忆检索 (Theory: 全模态拼接) ---
        # 拼接 F_d-seq, F_p-seq, F_d-struct, F_p-struct
        f_fused = torch.cat([d_vec, p_vec, s_vec, c_vec], dim=1) 
        m_vec = self.memory_bank(f_fused)

        # --- C. 预测 (Theory: 引力-斥力) ---
        # 引力关注结合潜力 (序列特征 + 经验)
        attr_input = torch.cat([d_vec, p_vec, m_vec], dim=1)
        attr = self.attraction_head(attr_input)
        
        # 斥力关注结构限制 (结构特征)
        repu_input = torch.cat([s_vec, c_vec], dim=1)
        repu = self.repulsion_head(repu_input)

        return attr - repu
