import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden1, hidden2, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.poo1=nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(embedding_dim, hidden1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden1, hidden1, kernel_size=3, stride=1, padding=1)
        self.poo2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden1, hidden2, kernel_size=3, stride=2, padding=1)
        self.poo3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.fc_mu = nn.Linear(hidden2*4, latent_dim)
        self.fc_logvar = nn.Linear(hidden2*4, latent_dim)

    def forward(self, x):
        # [B, 100, 9]
        B, N, C = x.shape
        x = self.embedding(x).permute(0, 2, 1)  # [B, embedding_dim, 100]
        x = self.poo1(x)            # [batch_size, embedding_dim, 50]
        x = F.relu(self.conv1(x))   # [batch_size, hidden1, 25]
        x = F.relu(self.conv2(x))   # [batch_size, hidden1, 25]
        x = self.poo2(x)            # [batch_size, hidden1, 13]
        x = F.relu(self.conv3(x))   # [batch_size, hidden2, 7]
        x = self.poo3(x)            # [batch_size, hidden2, 4]
        x = x.reshape(B ,-1)        # [batch_size, hidden2*4]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, kernel_size=3):
        super(Decoder, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size)
        self.deconv2 = nn.ConvTranspose1d(hidden_dim, embedding_dim, kernel_size)

    def forward(self, x):
        x = F.relu(self.fc(x)).unsqueeze(2)  # [batch_size, hidden_dim, 1]
        x = F.relu(self.deconv1(x))  # [batch_size, hidden_dim, kernel_size - 1]
        x = F.relu(self.deconv2(x))  # [batch_size, embedding_dim, kernel_size - 1 - kernel_size + 1]
        x = x.permute(0, 2, 1)  # [batch_size, kernel_size - 1 - kernel_size + 1, embedding_dim]
        return x

class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, kernel_size=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim, kernel_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim, kernel_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar



# Example usage:
if __name__ == '__main__':
    input = torch.ones(100, 100, 9).cuda()
    print(input.shape)

    encoder = Encoder(vocab_size=9, embedding_dim=128, hidden1=256, hidden2=512, latent_dim=1024).cuda()
    # decoder = Decoder(vocab_size=9, embedding_dim=1000, hidden_dim=1024, latent_dim=512, kernel_size=3).cuda()
    # model = VAE(vocab_size=9, embedding_dim=128, hidden_dim=1024, latent_dim=512, kernel_size=3).cuda()

    mu, logvar = encoder(input)
    print(mu.shape)
    print(summary(encoder,input_size=(100, 9) ))