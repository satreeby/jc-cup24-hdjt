import torch.nn as nn
import torch

class CpMLP(nn.Module):
    
    def __init__(self, inchans=6, hidden1=1024, hidden2=512, hidden3=64, outchans=5):
        super().__init__()
        self.Linear1=nn.Linear(inchans, hidden1)
        self.act=nn.Tanh()
        self.norm1=nn.BatchNorm1d(hidden1)
        self.Linear2=nn.Linear(hidden1, hidden2)
        self.norm2=nn.BatchNorm1d(hidden2)
        self.Linear3=nn.Linear(hidden2, hidden3)
        self.Linear4=nn.Linear(hidden3, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.act(self.Linear1(x))
        x=self.norm1(x)
        x=self.act(self.Linear2(x))
        x=self.norm2(x)
        x=self.act(self.Linear3(x))
        x=self.Linear4(x)
        return x
