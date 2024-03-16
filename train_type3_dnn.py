import torch
import numpy as np
from torch.utils import data
from model.type1 import CpMLP
from parser_work import parser_6_features

# 评价标准
def weighted_average_loss(outputs, labels):
    '''加权平均误差, [batch_size, 1] 的矩阵'''
    weights=torch.abs(labels)/(torch.abs(labels).sum(dim=1, keepdim=True))
    aveloss=torch.abs(outputs-labels)/torch.abs(labels)*weights
    return aveloss.sum(dim=1)

""" 
a=torch.tensor([[1,5,4]])
b=torch.tensor([[0,2,0]])
print(weighted_average_loss(b, a))
 """

# 获取参考电容
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
def load_data(labels_path, start, end):
    dataset_labels=np.array(read_data(labels_path))
    labels=torch.tensor(dataset_labels[start: end], dtype=torch.float32)
    return labels

labels_train = load_data('C:\project\python\HuadaJiutian\labels\labels_type2.txt', 0, 48)
labels_test = load_data('C:\project\python\HuadaJiutian\labels\labels_type3.txt', 0, 32)

# 获取输入数据

inputs_train=torch.tensor(parser_6_features.parser(type=2, Fpath='./data'),dtype=torch.float32)
inputs_test=torch.tensor(parser_6_features.parser(type=3, Fpath='./data'),dtype=torch.float32)

# 训练、测试模型
def train_model(model, criterion, optimizer, dataloader, features_train, labels_train):
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    average_loss = weighted_average_loss(model(features_train), labels_train).mean(dim=0)
    return average_loss

def test_model(model, features_test, labels_test):
    model.eval()
    test_loss= weighted_average_loss(model(features_test), labels_test)
    return test_loss


# 初始化模型、损失函数和优化器
batch_size=48
model = CpMLP.CpMLP(inchans=6, hidden1=128, hidden2=64, hidden3=32, outchans=4)
for param in model.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
for epoch in range(1000):
    train_loss = train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

# 测试
# 查看每个加权后的相对误差
model.eval()
print(f"predict: {model(inputs_test)}")
weights=torch.abs(labels_test)/(torch.abs(labels_test).sum(dim=1, keepdim=True))
print(f"weights: {weights}")
aveloss=torch.abs(model(inputs_test)-labels_test)/torch.abs(labels_test)*weights
print(f"Test Loss: {aveloss}")
# 总误差
print(f"Total Test Loss: {test_model(model, inputs_test, labels_test)}")