import torch.nn as nn
import torch
from torchsummary import summary


class PatcheEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embeded_dim=768, norm_layer=True):
        super().__init__()
        img_size=(img_size, img_size)
        patch_size=(patch_size, patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        self.num_patches=(img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])

        self.proj=nn.Conv2d(in_c, embeded_dim, patch_size, patch_size)
        self.norm=nn.LayerNorm(embeded_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W=x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # flatten: [B, C, H, W] => [B, C, HW]
        # transpose: [B, C, HW] => [B, HW, C]

        x=self.proj(x).flatten(2).transpose(1, 2)
        x=self.norm(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=None,
                 proj_drop_ratio=None) -> None:
        super().__init__()
        self.heads=heads
        # 每个头的维度
        head_dim=dim//heads
        # 默认为乘以根号下head_dim分之一
        self.qk_scale=qk_scale or head_dim**-0.5
        # 线性投影到qkv，故维度×3
        self.qkv=nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop_ratio)
        # 最后一层linear层
        self.proj=nn.Linear(dim, dim)
        self.proj_drop=nn.Dropout(proj_drop_ratio)
        
    def forward(self, x):
        '''input: [B, N, C]'''
        B, N, C=x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv=self.qkv(x).reshape(B, N, 3, self.heads, C//self.heads).permute(2, 0, 3, 1, 4)
        q, k, v= qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn=(q @ k.transpose(-2, -1))*self.qk_scale
        # 对最后一个维度做softmax
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x=(attn @ v).transpose(1, 2).reshape(B, N, C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 mlp_ratios=4,
                 act_layer=nn.LeakyReLU,
                 norm_layer=nn.BatchNorm1d) -> None:
        super().__init__()
        self.norm1=norm_layer(dim)
        self.attn=MultiHeadAttention(dim, heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.norm2=norm_layer(dim)
        self.ffn=MLP(dim, dim*mlp_ratios, act_layer, proj_drop_ratio)

    def forward(self, x):
        x=x+self.attn(self.norm1(x.transpose(2, 1)).transpose(2, 1))
        x=x+self.ffn(self.norm2(x.transpose(2, 1)).transpose(2, 1))
        return x

class CpT(nn.Module):
    def __init__(self, *, embeded_dim, channels=9, num_tokens=100,
                 heads=4, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d, depth=12,
                 num_classes=3, batch_size=259) -> None:
        super().__init__()
        # num_patches=(img_size//patch_size)*(img_size//patch_size)
        # self.patch_embeded=PatcheEmbed(img_size, patch_size, channels, embeded_dim)
        self.embeded=nn.Linear(channels, embeded_dim)
        # 电容词,nn.Parameter可学习,最后对其切片预测
        self.cls_token=nn.Parameter(torch.zeros(batch_size, 3, embeded_dim))
        # 位置编码
        # self.pos_embedded=nn.Parameter(torch.zeros(batch_size, num_patches+1, embeded_dim))
        # N个block, 可以用 *[×× for i in range()]，也可以用 nn.ModuleList
        self.blocks=nn.Sequential( 
            *[Block(embeded_dim, heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., 
                  mlp_ratios=4, act_layer=act_layer, norm_layer=norm_layer)
                  for i in range(depth)]
         )
        # 分类器
        self.mlp_head = nn.Linear(num_tokens*embeded_dim, num_classes)
        
    def forward(self, x):
        # x=self.patch_embeded(x)
        B, N, C=x.shape
        x=self.embeded(x)
        # x=torch.cat((self.cls_token, x), dim=1)
        # x=x+self.pos_embedded
        x=self.blocks(x)
        # 切片预测
        # x=x[:,0]
        x=x.view(B, -1)
        # 分类
        x=self.mlp_head(x)
        return x

    
if __name__ == '__main__':
    input = torch.ones(1, 100, 9).cuda()
    print(input.shape)
    model = CpT(embeded_dim=300, channels=9, heads=4, depth=1, 
                act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d, num_classes=3, batch_size=1).cuda()
    output = model(input)
    print(output.shape)
    print(summary(model,input_size=(100, 9)))