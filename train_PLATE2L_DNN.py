import torch
import numpy as np
from torch.utils import data
from model.PLATE2L import CpMLP
from parser_work import parser
from utils import cap_utils

device = torch.device('cuda:0')

"""加载训练，验证数据集以及标签"""

n_total = 972
n_train = 777
train_index = np.array(cap_utils.getRandomIndex(n_total=n_total, n_sample=n_train))
test_index = np.delete(np.arange(n_total), train_index)

labels_metal1=cap_utils.load_data('./Cases/PLATE2L/SUB-metal1_PLATE2L/SUB-metal1_PLATE2L.tbl.text', 0, 324)
labels_metal2=cap_utils.load_data('./Cases/PLATE2L/SUB-metal2_PLATE2L/SUB-metal2_PLATE2L.tbl.text', 0, 324)
labels_metal3=cap_utils.load_data('./Cases/PLATE2L/SUB-metal3_PLATE2L/SUB-metal3_PLATE2L.tbl.text', 0, 324)
labels=torch.cat((labels_metal1, labels_metal2, labels_metal3))
labels_train = labels[train_index, :].cuda()
labels_test = labels[test_index, :]

inputs_metal1=torch.tensor(parser.parser(pattern='PLATE2L', metal='1', pattern_path='./Cases'),dtype=torch.float32)
inputs_metal2=torch.tensor(parser.parser(pattern='PLATE2L', metal='2', pattern_path='./Cases'),dtype=torch.float32)
inputs_metal3=torch.tensor(parser.parser(pattern='PLATE2L', metal='3', pattern_path='./Cases'),dtype=torch.float32)
inputs=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3))
inputs_train=inputs[train_index, :].cuda()
inputs_test=inputs[test_index, :]

file = open("./log/PLATE2L/train_index.txt", "w+")
file.write(str(train_index))
file.close()
file = open("./log/PLATE2L/eval_index.txt", "w+")
file.write(str(test_index))
file.close()


"""训练、测试模型"""

# 初始化模型、损失函数和优化器
batch_size=259
model = CpMLP.CpMLP(inchans=14, hidden1=500, hidden2=300, hidden3=200, hidden4=200, outchans=3)
for param in model.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
for epoch in range(10000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    if train_loss<0.015:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

# 验证并记录
model.eval().to('cpu')
predict=model(inputs_test)
weights=torch.abs(labels_test)/(torch.abs(labels_test).sum(dim=1, keepdim=True))
aveloss=torch.abs(model(inputs_test)-labels_test)/torch.max(torch.abs(labels_test), torch.tensor(0.00001))*weights
test_error=cap_utils.test_model(model, inputs_test, labels_test, device='cpu')
train_error=cap_utils.test_model(model, inputs_train.cpu(), labels_train.cpu(), device='cpu')

print(f"Total Test Loss: {test_error}")
print(f"Total Test Loss: {test_error.mean()}")
file = open("./log/PLATE2L/results.txt", "w+")
file.write(f"predict\n {predict.detach().numpy()}\n Test Loss for every cap\n {aveloss.detach().numpy()}\n Test Loss\n {test_error.detach().numpy()}\n Average Test Loss\n {test_error.mean()}\n Train Loss\n {(train_error.detach().numpy())}")
file.close()

# 保存模型
torch.save(model.state_dict(), './model/PLATE2L/model_PLATE2L_CpMLP.pt')