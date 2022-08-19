import numpy as np
from sklearn.metrics import f1_score,fbeta_score,matthews_corrcoef,mean_squared_error

def calc_metrics(ypred,ytrue):
    """
    Calculates f1 & f2 scores, MCC, and mean-squared-error

    Args:
        ypred (np.ndarray): prediction labels
        ytrue (np.ndarray): true labels

    Returns:
        f1 [float], f2[float], mcc[float], mse[float]
    """
    ypred = np.array(ypred)
    print('Calculating Predictions...')
    preds,lbls = ypred.argmax(-1),ytrue.argmax(-1)
    print('Calculating F1...')
    f1 = f1_score(lbls,preds,average='macro')
    print('Calculating F2...')
    f2 = fbeta_score(lbls,preds,beta=2,average='macro')
    print('Calculating MCC...')
    mcc = matthews_corrcoef(lbls,preds)
    print('Calculating MSE...')
    mse = mean_squared_error(lbls,preds)
    return f1,f2,mcc,mse




