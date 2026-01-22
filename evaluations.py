import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_rm2(y_true, y_pred):
    """
    计算 Modified Squared Correlation Coefficient (rm^2)。
    这是 DTA 任务中用于衡量预测值与观察值一致性的重要指标。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算标准相关系数 r^2
    r2 = pearsonr(y_true, y_pred)[0] ** 2
    
    # 计算通过原点的相关系数 r0^2
    # 这里的 k 是回归线的斜率
    k = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    res_sum = np.sum((y_true - k * y_pred) ** 2)
    tot_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    r02 = 1 - res_sum / (tot_sum + 1e-10) # 防止除以零
    
    # rm2 = r2 * (1 - sqrt(r2 - r02))
    rm2 = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
    return rm2

def get_regression_metrics(y_true, y_pred):
    """返回所有回归评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson = pearsonr(y_true, y_pred)[0]
    ci = concordance_index(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred) # 计算 rm2
    
    return mse, rmse, pearson, ci, rm2

def concordance_index(y_true, y_pred):
    """计算一致性指数 (Concordance Index)"""
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    i = len(y_true)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z += 1.0
                if y_pred[i] > y_pred[j]: S += 1.0
                elif y_pred[i] == y_pred[j]: S += 0.5
            j -= 1
        i -= 1
        j = i-1
    return S/z if z > 0 else 0.5

def get_mean_and_std(data_list):
    """计算均值和标准差"""
    return np.mean(data_list), np.std(data_list)