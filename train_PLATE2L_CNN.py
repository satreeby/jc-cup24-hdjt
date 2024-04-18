import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from model.PLATE2L import CpMLP, CpTransformer, CpCNN
from parser_work import parser_coordinates
import matplotlib.pyplot as plt
device = torch.device('cuda:0')


# 评价标准
def weighted_average_loss(outputs, labels):
    '''加权平均误差, [batch_size, 1] 的矩阵'''
    weights=torch.abs(labels)/(torch.abs(labels).sum(dim=1, keepdim=True))
    aveloss=torch.abs(outputs-labels)/ torch.max(torch.abs(labels), torch.tensor(0.00001))*weights
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


def getRandomIndex(n, x):
	# 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index
n=324
x=259
train_index = np.array(getRandomIndex(n, x))
test_index = np.delete(np.arange(n), train_index)
file = open(".\\log\\pattern1_metal_train.txt", "w+")
content = str(train_index)
file.write(content)
file.close()
file = open(".\\log\\pattern1_metal_test.txt", "w+")
content = str(test_index)
file.write(content)
file.close()

labels_metal1=load_data('.\Cases\PLATE2L\SUB-metal1_PLATE2L\SUB-metal1_PLATE2L.tbl.text', 0, 324)
labels_metal2=load_data('.\Cases\PLATE2L\SUB-metal2_PLATE2L\SUB-metal2_PLATE2L.tbl.text', 0, 324)
labels_metal3=load_data('.\Cases\PLATE2L\SUB-metal3_PLATE2L\SUB-metal3_PLATE2L.tbl.text', 0, 324)
labels_train = torch.cat((labels_metal1[train_index, :], labels_metal2[train_index, :], labels_metal3[train_index, :])).cuda()
labels_test = torch.cat((labels_metal1[test_index, :], labels_metal2[test_index, :], labels_metal3[test_index, :]))
# 获取输入数据
# metal1数据
inputs_metal1=torch.tensor(parser_coordinates.parser(pattern='PLATE2L', metal='1', pattern_path='.\Cases\PLATE2L'),dtype=torch.float32)

# metal2数据
inputs_metal2=torch.tensor(parser_coordinates.parser(pattern='PLATE2L', metal='2', pattern_path='.\Cases\PLATE2L'),dtype=torch.float32)

# metal3数据
inputs_metal3=torch.tensor(parser_coordinates.parser(pattern='PLATE2L', metal='3', pattern_path='.\Cases\PLATE2L'),dtype=torch.float32)

inputs_train=torch.cat((inputs_metal1[train_index, :], inputs_metal2[train_index, :], inputs_metal3[train_index, :])).cuda()
inputs_test=torch.cat((inputs_metal1[test_index, :], inputs_metal2[test_index, :], inputs_metal3[test_index, :]))

print(inputs_train.shape)


# 训练、测试模型
def train_model(model, criterion, optimizer, dataloader, features_train, labels_train):
    model
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
    average_loss = weighted_average_loss(model(features_train), labels_train).max()
    return average_loss

def test_model(model, features_test, labels_test):
    model=model.cpu()
    model.eval()
    features_test=features_test
    labels_test=labels_test
    test_loss= weighted_average_loss(model(features_test), labels_test)
    return test_loss


# 初始化模型、损失函数和优化器
batch_size=777
# model = CpTransformer.CpT(embeded_dim=100, channels=9, heads=4, depth=3, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d, num_classes=3).cuda()
# model = CpMLP.CpMLP_Mixer(inchans=9, embedded_dim=128, num_tokens=100, hidden1=512, hidden2=512, outchans=3).cuda()
model = CpCNN.CPCNN(vocab_size=9, embedding_dim=128, hidden1=256, hidden2=512, hidden3=512, hidden4=512, hidden5=256, out_chans=3).cuda()
for param in model.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
# x=[]
# y=[]
for epoch in range(10000):
    train_loss = train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train)
    # if epoch%10 == 0:
    #     x.append(epoch)
    #     y.append(train_loss)
    if train_loss<0.03:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

# 测试
# 查看每个加权后的相对误差

model.eval()
model=model.cpu()
predict=model(inputs_test)
# print(f"predict: {model(inputs_test)}")
weights=torch.abs(labels_test)/(torch.abs(labels_test).sum(dim=1, keepdim=True))
# print(f"weights: {weights}")
aveloss=torch.abs(model(inputs_test)-labels_test)/torch.max(torch.abs(labels_test), torch.tensor(0.00001))*weights
# print(f"Test Loss for every cap: {aveloss}")
# 总误差
test_error=test_model(model, inputs_test, labels_test)
train_error=test_model(model, inputs_train.cpu(), labels_train.cpu())
print(f"Total Test Loss: {test_error}")
# 平均误差
print(f"Total Test Loss: {test_error.mean()}")


file = open(".\\log\\pattern1_metal_attention_results.txt", "w+")
file.write(f"predict\n {predict.detach().numpy()}\n Test Loss for every cap\n {aveloss.detach().numpy()}\n Test Loss\n {test_error.detach().numpy()}\n Average Test Loss\n {test_error.mean()}\n Train Loss\n {(train_error.detach().numpy())}")
file.close()