from typing import Any
import torch
import numpy as np
import os

class EarlyStopping():
    def __init__(self, verbose=False, patience=15, delta=0.06) -> None:
        """
        Args:
        verbose: True 输出val_loss下降的improvement
        patience: eval_loss上升后n个epoch停止
        delta: 最小变化
        """
        self.verbose = verbose
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, train_loss, model):
        score = abs(train_loss - val_loss)
        
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
        elif score > self.best_score + self.delta: # evalloss不再下降
            print("")
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
            self.count = 0
