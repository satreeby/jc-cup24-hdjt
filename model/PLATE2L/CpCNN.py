import torch.nn as nn
import torch
from torchsummary import summary

class CPCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden1, hidden2, hidden3, hidden4, hidden5, out_chans):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.embedding = nn.Conv1d(vocab_size, embedding_dim, kernel_size=1, padding=0, stride=1)
        self.norm0=nn.BatchNorm1d(embedding_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(embedding_dim, hidden1, kernel_size=3, stride=2, padding=1)
        self.norm1=nn.BatchNorm1d(hidden1)
        self.conv2 = nn.Conv1d(hidden1, hidden1, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(hidden1, hidden1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(hidden1, hidden1, kernel_size=3, stride=1, padding=1)
        self.norm2=nn.BatchNorm1d(hidden1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden1, hidden2, kernel_size=3, stride=2, padding=1)
        self.norm3=nn.BatchNorm1d(hidden2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.linear1 = nn.Linear(hidden2*4, hidden3)
        self.linear2 = nn.Linear(hidden3, hidden4)
        self.linear3 = nn.Linear(hidden4, hidden5)
        self.linear4 = nn.Linear(hidden5, out_chans)

    def forward(self, x):
        # [B, 100, 9]
        x = x.to(torch.float32)
        B, N, C = x.shape
        x = self.embedding(x.permute(0, 2, 1))  # [B, embedding_dim, 100]
        x = self.pool1(x)            # [batch_size, embedding_dim, 50]
        x = self.norm0(x)
        x = self.act(self.conv1(x))   # [batch_size, hidden1, 25]
        x = self.norm1(x)
        x = self.act(self.conv2(x))   # [batch_size, hidden1, 25]
        x = self.norm2(x)
        x = x+self.act(self.conv5(self.act(self.conv4(x))))   # [batch_size, hidden1, 25]
        x = self.pool2(x)            # [batch_size, hidden1, 13]
        x = self.act(self.conv3(x))   # [batch_size, hidden2, 7]
        x = self.norm3(x)
        x = self.pool3(x)            # [batch_size, hidden2, 4]
        x = x.reshape(B ,-1)        # [batch_size, hidden2*4]
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.linear4(x)

        return x


if __name__ == '__main__':
    input = torch.ones(100, 100, 9).cuda()
    print(input.shape)
    model = CPCNN(vocab_size=9, embedding_dim=128, hidden1=256, hidden2=512, hidden3=512, hidden4=512, hidden5=256, out_chans=3).cuda()
    output = model(input)
    print(output.shape)
    print(summary(model,input_size=(100, 9) ))