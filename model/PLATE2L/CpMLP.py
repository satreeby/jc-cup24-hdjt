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

class CpMLP_Mixer(nn.Module):
    def __init__(self, inchans=6, embedded_dim=64, num_tokens=100, hidden1=512, hidden2=512, outchans=3):
        super().__init__()
        # [B, C, N], N=100
        self.embedded=nn.Linear(inchans, embedded_dim)
        self.norm0=nn.BatchNorm1d(embedded_dim)
        self.act=nn.LeakyReLU()

        self.linear1=nn.Linear(embedded_dim, hidden1)
        self.linear2=nn.Linear(hidden1, embedded_dim)

        self.linear3=nn.Linear(num_tokens, hidden2)
        self.linear4=nn.Linear(hidden2, num_tokens)

        self.pool=nn.AdaptiveAvgPool1d(1)
        self.linear5=nn.Linear(embedded_dim, outchans)

    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.embedded(x)
        x=self.norm0(x.transpose(1,2)).transpose(1,2)

        x=x+self.linear2(self.act(self.linear1(x)))
        x=self.norm0(x.transpose(1,2)).transpose(1,2)
        x=x.transpose(1, 2)
        x=x+self.linear4(self.act(self.linear3(x)))
        x=x.transpose(1, 2)
        x=self.norm0(x.transpose(1,2)).transpose(1,2)

        x=x+self.linear2(self.act(self.linear1(x)))
        x=self.norm0(x.transpose(1,2)).transpose(1,2)
        x=x.transpose(1, 2)
        x=x+self.linear4(self.act(self.linear3(x)))
        x=x.transpose(1, 2)
        x=self.norm0(x.transpose(1,2)).transpose(1,2)

        x=x+self.linear2(self.act(self.linear1(x)))
        x=self.norm0(x.transpose(1,2)).transpose(1,2)
        x=x.transpose(1, 2)
        x=x+self.linear4(self.act(self.linear3(x)))
        x=x.transpose(1, 2)
        x=self.norm0(x.transpose(1,2)).transpose(1,2)

        x=x+self.linear2(self.act(self.linear1(x)))
        x=self.norm0(x.transpose(1,2)).transpose(1,2)
        x=x.transpose(1, 2)
        x=x+self.linear4(self.act(self.linear3(x)))
        x=x.transpose(1, 2)
        x=self.norm0(x.transpose(1,2)).transpose(1,2)


        x=self.pool(x.transpose(1,2)).transpose(1,2)
        x=self.linear5(x.squeeze(1))

        return x


        
        


if __name__ == '__main__':
    input = torch.ones(100, 100, 9).cuda()
    print(input.shape)
    model = CpMLP_Mixer(inchans=9, embedded_dim=64, num_tokens=100, hidden1=256, hidden2=512, outchans=3).cuda()
    output = model(input)
    print(output.shape)
    print(summary(model,input_size=(100, 9) ))

if __name__ == '__main__':
    model = CpMLP(inchans=15, hidden1=500, hidden2=300, hidden3=200, hidden4=200, outchans=3).cuda()
    print(summary(model,input_size=(15, ) ))