import torch.nn as nn
import torch

class CpMLP(nn.Module):
    
    def __init__(self, inchans=6, hidden1=512, hidden2=128, outchans=5):
        super().__init__()
        self.Linear1=nn.Linear(inchans, hidden1)
        self.act=nn.Tanh()
        self.norm1=nn.BatchNorm1d(hidden1)
        self.Linear2=nn.Linear(hidden1, hidden2)
        self.Linear3=nn.Linear(hidden2, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.act(self.Linear1(x))
        x=self.norm1(x)
        x=self.act(self.Linear2(x))
        x=self.Linear3(x)
        return x
    
class CpTokenMLP(nn.Module):
    
    def __init__(self, inchans=4, hiddens=512, outchans=5):
        super().__init__()
        self.Linear1=nn.Linear(inchans, hiddens)
        self.act=nn.Tanh()
        self.norm1=nn.BatchNorm1d(hiddens)
        self.Linear2=nn.Linear(hiddens, outchans)
        self.norm2=nn.BatchNorm1d(outchans)

    
    def forward(self, x):
        '''input: B, C, N'''
        x=x.to(torch.float32).permute(0, 2, 1)
        x=self.act(self.Linear1(x))
        x=self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x=self.Linear2(x)
        x=self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1).mean(dim=1)
        return x

