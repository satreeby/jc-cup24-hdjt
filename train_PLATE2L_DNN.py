import torch
import numpy as np
from torch.utils import data
from model.PLATE2L import CpMLP
from parser_work import parser_1k8
from utils import cap_utils
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda:0')

"""加载训练，验证数据集以及标签"""

n_total = 1944
n_train = 1555
train_index = np.array(cap_utils.getRandomIndex(n_total=n_total, n_sample=n_train))
test_index = np.delete(np.arange(n_total), train_index)

""" matrix = []
filename = './log/PLATE2L/val_index.txt'
with open(filename, 'r') as file:
    for line in file:
        matrix.extend(map(int, line.strip().split()))

test_index=np.array(matrix)
train_index=np.delete(np.arange(n_total), test_index) """


labels_metal1=cap_utils.load_data('./Cases/PLATE2L/SUB-metal1_PLATE2L/SUB-metal1_PLATE2L.tbl.text', 0, 324)
labels_metal2=cap_utils.load_data('./Cases/PLATE2L/SUB-metal2_PLATE2L/SUB-metal2_PLATE2L.tbl.text', 0, 324)
labels_metal3=cap_utils.load_data('./Cases/PLATE2L/SUB-metal3_PLATE2L/SUB-metal3_PLATE2L.tbl.text', 0, 324)
labels_metal456=cap_utils.load_data('./Cases/PLATE2L/SUB-metal456_PLATE2L/SUB-metal456_PLATE2L.tbl.text', 0, 972)
# labels_metal_extention1=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1/final_output.txt', 0, 726)
# labels_metal_extention2=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1c2/final_output.txt', 0, 13794)
# labels_metal_extention3=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_4model_c1c2lEnvrEnv/final_output.txt', 0, 46170)
# labels_metal_extention4=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1lEnv/final_output.txt', 0, 3630)
# labels_metal_extentions=cap_utils.load_data('./Cases/PLATE2L/final_output.txt', 0, 4320)[:, [0, 2, 4]]
# labels_extentions=cap_utils.load_data('./Extentions/PLATE2L/PLATE2L_random_files.tbl.txt', 0, 24388)[:, [0, 2, 4]]

# labels=torch.cat((labels_metal1, labels_metal2, labels_metal3, labels_metal456))
# labels=torch.cat((labels_metal_extention1, labels_metal_extention2, labels_metal_extention3, labels_metal_extention4))[:, [1, 3, 5]]
# labels=torch.cat((labels, labels_metal456))
labels_metal=torch.cat((labels_metal1, labels_metal2, labels_metal3))
# labels_train=labels_metal[train_index,:].cuda()
# labels_test=labels_metal[test_index,:]
labels_train = labels_metal.cuda()
labels_test = labels_metal456

print(labels_train.shape)

inputs_metal1=torch.tensor(parser_1k8.parser(pattern='PLATE2L', metal='1', pattern_path='./Cases', gen=False, input_num=324),dtype=torch.float32)
inputs_metal2=torch.tensor(parser_1k8.parser(pattern='PLATE2L', metal='2', pattern_path='./Cases', gen=False, input_num=324),dtype=torch.float32)
inputs_metal3=torch.tensor(parser_1k8.parser(pattern='PLATE2L', metal='3', pattern_path='./Cases', gen=False, input_num=324),dtype=torch.float32)
inputs_metal456=torch.tensor(parser_1k8.parser(pattern='PLATE2L',  metal='3', pattern_path='./Cases/PLATE2L/SUB-metal456_PLATE2L/input', gen=True, input_num=972),dtype=torch.float32)
# inputs_metal_extentions1=torch.tensor(parser_1k8_2.parser(pattern='PLATE2L',  metal='3', pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1/PLATE2L_random_file', gen=True, input_num=726),dtype=torch.float32)
# inputs_metal_extentions2=torch.tensor(parser_1k8_2.parser(pattern='PLATE2L',  metal='3', pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1c2/PLATE2L_random_file', gen=True, input_num=13794),dtype=torch.float32)
# inputs_metal_extentions3=torch.tensor(parser_1k8_2.parser(pattern='PLATE2L',  metal='3', pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_random_file', gen=True, input_num=46170),dtype=torch.float32)
# inputs_metal_extentions4=torch.tensor(parser_1k8_2.parser(pattern='PLATE2L',  metal='3', pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1lEnv/PLATE2L_random_file', gen=True, input_num=3630),dtype=torch.float32)
# inputs=torch.cat((inputs_metal_extentions1, inputs_metal_extentions2, inputs_metal_extentions3, inputs_metal_extentions4))
# inputs=torch.cat((inputs, inputs_metal456))
inputs_metal=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3))

# inputs_metal_extentions=torch.tensor(parser_extend_1.parser(pattern='PLATE2L', input_num=4320, pattern_path='./Cases/PLATE2L/PLATE2L_random_files'),dtype=torch.float32)


# inputs=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3, inputs_metal456))
# inputs_train=inputs_metal[train_index,:].cuda()
# inputs_test=inputs_metal[test_index,:]
inputs_train=inputs_metal.cuda()
inputs_test=inputs_metal456

print(inputs_train.shape)

file = open("./log/PLATE2L/train_index.txt", "w+")
file.write(str(train_index))
file.close()
file = open("./log/PLATE2L/val_index.txt", "w+")
file.write(str(test_index))
file.close()


"""训练、测试模型"""

# 初始化模型、损失函数和优化器
batch_size=4096
model = CpMLP.CpMLP(inchans=26, hidden1=4096, hidden2=1024, hidden3=512, hidden4=128, hidden5=128, outchans=3)
# model = CpTransformer.TransformerModel(inchans=26, hidden=1024, outchans= 3).cuda()
# for param in model.parameters():
#    torch.nn.init.normal_(param, mean=0, std=0.01)
state_dict = torch.load('./model/PLATE2L/saved/model_PLATE2L_CpMLP_expand_80percent.pt')
model.load_state_dict(state_dict)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1 ,weight_decay=1e-8)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
from torch.optim.lr_scheduler import LambdaLR
scheduler_1 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.001)
scheduler_2 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.0001)

""" for epoch in range(20000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
    if train_loss<0.01:
        break """

""" scheduler_1.step()
print(f"new lr = {optimizer.param_groups[0]['lr']}")

for epoch in range(40000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    if train_loss<0.03:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

scheduler_2.step()
print(f"new lr = {optimizer.param_groups[0]['lr']}")

for epoch in range(40000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    if train_loss<0.01:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}") """

# 保存模型
# torch.save(model.state_dict(), './model/PLATE2L/model_PLATE2L_CpMLP.pt')

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
