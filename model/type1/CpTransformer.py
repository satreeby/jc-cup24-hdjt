import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, N]
    Output: tensor in shape [B, C, N]
    """
    def __init__(self, kernel_size=1, stride=1, padding=0, 
                 in_chans=4, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    

class Mlp(nn.Module):
    """
    Input: tensor in shape [B, N, C]
    Output: tensor in shape [B, N, C]
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Tanh, norm_layer=nn.BatchNorm1d, drop=0.01):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Input: tensor in shape [B, N, C]
    Output: tensor in shape [B, N, C]
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.01):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Cpformer(nn.Module):

    def __init__(self, in_chans=4, out_chans=5, kernel_size=1, stride=1, padding=0,
                 dim=1024, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.01,
                 act_layer=nn.Tanh, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.patchembed = PatchEmbed(
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            in_chans=in_chans, 
            embed_dim=dim
        )
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cp = nn.Linear(in_features=dim, out_features=out_chans)

    def forward(self, x):
        x=x.to(torch.float32)
        x = self.patchembed(x).permute(0, 2, 1)
        x = self.attn(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.mlp(x)
        x = self.cp(x).mean(dim=1)

        return x

""" conductor = torch.ones((5, 4, 4))
net = Cpformer(in_chans=4, out_chans=5, dim=512, num_heads=8, mlp_ratio=4, 
               qk_scale=None, drop=0, attn_drop=0, 
               act_layer=nn.Tanh, norm_layer=nn.BatchNorm1d)

capacitor=net(conductor) """