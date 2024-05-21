import torch.nn as nn
import torch
from torchsummary import summary

class CpMLP(nn.Module):
    
    def __init__(self, inchans=6, hidden1=1024, hidden2=512, hidden3=64, hidden4=100, hidden5=64, outchans=5):
        super().__init__()
        self.Linear1=nn.Linear(inchans, hidden1)
        self.act1=nn.LeakyReLU()
        self.act2=nn.Tanh()
        self.norm1=nn.BatchNorm1d(hidden1)
        self.Linear2=nn.Linear(hidden1, hidden2)
        self.norm2=nn.BatchNorm1d(hidden2)
        self.Linear3=nn.Linear(hidden2, hidden3)
        self.norm3=nn.BatchNorm1d(hidden3)
        self.Linear4=nn.Linear(hidden3, hidden4)
        self.Linear5=nn.Linear(hidden4, hidden5)
        self.Linear6=nn.Linear(hidden5, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.act1(self.Linear1(x))
        x=self.norm1(x)
        x=self.act1(self.Linear2(x))
        x=self.norm2(x)
        x=self.act1(self.Linear3(x))
        x=self.norm3(x)
        x=self.act1(self.Linear4(x))
        x=self.act1(self.Linear5(x))
        x=self.Linear6(x)
        return x