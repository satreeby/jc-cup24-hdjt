import torch
import numpy as np
from torch.utils import data
import read_data_2
from model.type1 import CpTransformer, CpMLP
import torch.nn as nn

# 生成数据集，读数
batch_size=8

def read_data(file_path):
    try:
        with open(file_path, 'r') as file:
            values_list=[]
            for line in file:
                values=[float(num) for num in line.split()]
                values_list.append(values)
            return values_list
    except FileNotFoundError:
        print("no such file")
        return None
    except Exception as e:
        print("error")
        return None

geology_tensor=read_data_2.load_data(64)
features_samples=geology_tensor[0: 48]
features_test=geology_tensor[48: 64]

def load_data(file_path, start, end):
    dataset=np.array(read_data(file_path))
    capacitance_tensor=torch.tensor(dataset[start: end, 6:12], dtype=torch.float32)
    return capacitance_tensor

labels_samples = load_data('C:\\project\\python\\HuadaJiutian\\test\\type1.text', 0, 48)
labels_test = load_data('C:\\project\\python\\HuadaJiutian\\test\\type1.text', 48, 64)

lambda1=0.01

# 定义训练函数
    
def train_model(model, criterion, optimizer, dataloader, train_indices):
    model.train()
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    running_loss = ((torch.abs((model(features_samples[train_indices])-labels_samples[train_indices])/labels_samples[train_indices]))).mean(dim=0)
    return running_loss

def test_model(model, dataloader):
    model.eval()
    val_loss= ((torch.abs((model(features_samples[val_indices])-labels_samples[val_indices])/labels_samples[val_indices]))).mean(dim=0)
    return val_loss 

# 划分数据集为 k 折
total_samples = 48
k = 3
fold_size = total_samples // k
fold_indices = [(i * fold_size, (i + 1) * fold_size) for i in range(k)]

# 初始化模型、损失函数和优化器
""" model = CpTransformer.Cpformer(in_chans=4, out_chans=5, dim=128, num_heads=4, mlp_ratio=2, 
           qk_scale=False, drop=0., attn_drop=0., 
           act_layer=nn.Tanh, norm_layer=nn.BatchNorm1d) """
model = CpMLP.CpTokenMLP(inchans=4, hiddens=512, outchans=5)
for param in model.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 打印每个折的训练和测试结果
for fold in range(k):
    print(f"Fold {fold + 1}:")
    # 划分数据集
    val_indices = list(range(*fold_indices[fold]))
    train_indices = [i for i in range(total_samples) if i not in val_indices]
    train_loader = data.DataLoader(data.TensorDataset(features_samples[train_indices], labels_samples[train_indices]), batch_size, shuffle=True)
    val_loader = data.DataLoader(data.TensorDataset(features_samples[val_indices], labels_samples[val_indices]), batch_size, shuffle=False)
    

    
    # 训练模型
    for epoch in range(1000):
        train_loss = train_model(model, criterion, optimizer, train_loader, train_indices)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
    
    # 测试模型
    accuracy = test_model(model, val_loader)

model.eval()
outputs_test = model(features_test)
loss=torch.abs(((outputs_test-labels_test)/labels_test)).mean(dim=0)
print(loss)