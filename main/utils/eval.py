import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.config import CONFIGCLASS
from utils.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy

        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    val_cfg.datasets = cfg.datasets_test
    val_cfg.isTrain = False
    # val_cfg.aug_resize = False
    # val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg

def validate(model: nn.Module, cfg: CONFIGCLASS):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score, roc_curve

    from utils.datasets import create_dataloader

    data_loader = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:

            dire, img, label = data
        
            dire = to_cuda(dire, device)
            img = to_cuda(img, device)       
                  
            predict = model(img, dire)
            
            y_pred.extend(predict.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    # Calculate F1, AUC, ROC using sklearn
    
    # F1 score
    f1 = f1_score(y_true, y_pred > 0.5)
    
    # AUC score
    auc_score = roc_auc_score(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    results = {
        "ACC": acc,
        "AP": ap,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
        "F1": f1,
        "AUC": auc_score,
        # "ROC": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    }
    return results
