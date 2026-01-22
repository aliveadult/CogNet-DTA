import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utilss import HGDDTIDataset, collate_fn_combined, load_data, get_k_fold_data
from models import CogNetDTA
from configss import Configs
from evaluations import get_regression_metrics, get_mean_and_std 

# ================= UQ (Uncertainty Quantization) 核心辅助函数 =================
def activate_dropout(model):
    """落实 MC Dropout：强制开启模型中所有的 Dropout 层"""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()

def evaluate_with_uq(model, loader, device, T=10):
    """
    执行 Monte Carlo Dropout 采样来计算预测值及其不确定性。
    T: 采样次数（为了兼顾训练速度，Epoch 期间建议设为较小值，如 5-10）
    """
    model.eval()
    activate_dropout(model) 
    
    all_true = []
    all_preds_samples = [] 

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            g, d, p, a, c = batch
            if g is None: continue
            g, d, p, a = g.to(device), d.to(device), p.to(device), a.to(device)
            c = [m.to(device) for m in c]
            
            batch_samples = []
            for _ in range(T):
                out, _, _ = model(g, d, p, c)
                batch_samples.append(out.cpu().numpy().flatten())
            
            all_preds_samples.append(np.stack(batch_samples, axis=0)) 
            all_true.extend(a.cpu().numpy().flatten())

    all_preds_samples = np.concatenate(all_preds_samples, axis=1)
    final_preds = np.mean(all_preds_samples, axis=0)
    uncertainties = np.std(all_preds_samples, axis=0)
    
    mse, rmse, pearson, ci, rm2 = get_regression_metrics(all_true, final_preds)
    
    return {
        'MSE': mse, 'RMSE': rmse, 'Pearson': pearson, 'CI': ci, 'RM2': rm2,
        'UQ': np.mean(uncertainties) 
    }

# ================= 训练模块 =================
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        g, d, p, a, c = batch
        if g is None: continue
        g, d, p, a = g.to(device), d.to(device), p.to(device), a.to(device)
        c = [m.to(device) for m in c]
        optimizer.zero_grad()
        out, _, _ = model(g, d, p, c)
        loss = criterion(out, a)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    config = Configs()
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)
    df, esm_embeddings = load_data(config)
    k_folds = get_k_fold_data(df, config.n_splits, config.random_state)
    final_metrics = []

    for fold, (train_df, test_df) in enumerate(k_folds):
        print(f"\n>>> Fold {fold+1} | CogNet-DTA Start")
        model = CogNetDTA(config.drug_fp_size, config).to(device)
        
        criterion = nn.MSELoss() 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        train_loader = DataLoader(HGDDTIDataset(train_df, esm_embeddings, config, config.contact_map_dir), 
                                  batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_combined)
        test_loader = DataLoader(HGDDTIDataset(test_df, esm_embeddings, config, config.contact_map_dir), 
                                 batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_combined)

        best_mse = float('inf')
        for epoch in range(1, config.epochs + 1):
            _ = train(model, train_loader, optimizer, criterion, device)
            # 每个 epoch 调用 evaluate_with_uq 以获取 UQ 指标
            # 这里设置 T=5 以在保证 UQ 采样的同时不严重拖慢训练速度
            m = evaluate_with_uq(model, test_loader, device, T=5)
            scheduler.step()
            
            # --- 优化后的缩写输出：移除 Loss/RMSE，增加 UQ ---
            print(f"Epoch {epoch:03d} | MSE: {m['MSE']:.4f} | Pearson: {m['Pearson']:.4f} | CI: {m['CI']:.4f} | RM2: {m['RM2']:.4f} | UQ: {m['UQ']:.4f}")
            
            if m['MSE'] < best_mse:
                best_mse = m['MSE']
                torch.save(model.state_dict(), os.path.join(config.output_dir, f'best_fold_{fold+1}.pt'))
        
        # 加载最优权重进行 Fold 最终采样 (T=20)
        model.load_state_dict(torch.load(os.path.join(config.output_dir, f'best_fold_{fold+1}.pt')))
        final_metrics.append(evaluate_with_uq(model, test_loader, device, T=20))

    # --- 最终 K-Fold 汇总报告 (根据之前记忆，此处保持指标全称及 00.00 格式) ---
    print("\n" + "="*95)
    print(f"{'CogNet-DTA Final K-Fold Summary Report (with UQ)':^95}")
    print("="*95)
    metrics_to_show = [
        ('MSE', 'Mean Squared Error'), 
        ('Pearson', 'Pearson Correlation Coefficient'), 
        ('CI', 'Concordance Index'), 
        ('RM2', 'Modified Squared Correlation Coefficient'),
        ('UQ', 'Mean Uncertainty (Standard Deviation)')
    ]
    
    for key, full_name in metrics_to_show:
        vals = [m[key] for m in final_metrics]
        mean_v, std_v = np.mean(vals), np.std(vals)
        print(f"{full_name:<55} | {mean_v:05.2f} ± {std_v:05.2f}")
    print("="*95)

if __name__ == '__main__':
    main()