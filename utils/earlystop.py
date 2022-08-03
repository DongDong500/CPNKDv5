#import EarlyStopping
#from pytorchtools import EarlyStopping
import numpy as np
import torch
import os

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience:int = 7, verbose:bool = False, delta:int = 0, 
                    path:str = 'checkpoint.pt', save_model:bool = False):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save_model = save_model

    def __call__(self, val_loss, model, optim, scheduler, cur_itrs):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, scheduler, cur_itrs)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, scheduler, cur_itrs)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model, optim, scheduler, cur_itrs):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f})')
        if self.save_model:
            torch.save({
                'model_state' : model.state_dict(),
                'optimizer_state' : optim.state_dict(),
                'scheduler_state' : scheduler.state_dict(),
                'cur_itrs' : cur_itrs,
            }, os.path.join(self.path, 'checkpoint.pt'))
        else:
            torch.save({
                'model_state' : model.state_dict(),
                'optimizer_state' : optim.state_dict(),
                'scheduler_state' : scheduler.state_dict(),
                'cur_itrs' : cur_itrs,
            }, os.path.join(self.path, 'checkpoint.pt'))

        self.val_loss_min = val_loss