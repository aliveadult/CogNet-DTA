import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_rm2(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    

    r2 = pearsonr(y_true, y_pred)[0] ** 2
    

    k = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    res_sum = np.sum((y_true - k * y_pred) ** 2)
    tot_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    r02 = 1 - res_sum / (tot_sum + 1e-10)
    
    # rm2 = r2 * (1 - sqrt(r2 - r02))
    rm2 = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
    return rm2

def get_regression_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson = pearsonr(y_true, y_pred)[0]
    ci = concordance_index(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    
    return mse, rmse, pearson, ci, rm2

def concordance_index(y_true, y_pred):

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
    return np.mean(data_list), np.std(data_list)
