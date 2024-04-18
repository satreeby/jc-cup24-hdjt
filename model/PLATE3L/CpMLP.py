import torch.nn as nn
import torch
from torchsummary import summary

class CpMLP(nn.Module):
    
    def __init__(self, inchans=6, hidden1=1024, hidden2=512, hidden3=64, hidden4=100, outchans=5):
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
        self.Linear5=nn.Linear(hidden4, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.act1(self.Linear1(x))
        x=self.norm1(x)
        x=self.act1(self.Linear2(x))
        x=self.norm2(x)
        x=self.act1(self.Linear3(x))
        x=self.norm3(x)
        x=self.act1(self.Linear4(x))
        x=self.Linear5(x)
        return x

class CpMLP_for_Cordinate(nn.Module):
    
    def __init__(self, inchans=900, hidden1=1024, hidden2=512, hidden3=64, hidden4=100, outchans=5):
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
        self.Linear5=nn.Linear(hidden4, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=x.flatten(start_dim=1, end_dim=-1)
        x=self.act1(self.Linear1(x))
        x=self.norm1(x)
        x=self.act1(self.Linear2(x))
        x=self.norm2(x)
        x=self.act1(self.Linear3(x))
        x=self.norm3(x)
        x=self.act1(self.Linear4(x))
        x=self.Linear5(x)
        return x

if __name__ == '__main__':
    model = CpMLP(inchans=15, hidden1=500, hidden2=300, hidden3=200, hidden4=200, outchans=3).cuda()
    print(summary(model,input_size=(15, ) ))