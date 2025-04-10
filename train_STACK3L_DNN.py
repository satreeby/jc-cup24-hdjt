import torch
import numpy as np
from torch.utils import data
from model.STACK3L import CpMLP
from parser_work import parser_single_expand
from utils import cap_utils
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

device = torch.device('cuda:0')

"""加载训练，验证数据集以及标签"""

n_total = 432
n_train = 345
train_index = np.array(cap_utils.getRandomIndex(n_total=n_total, n_sample=n_train))
test_index = np.delete(np.arange(n_total), train_index)

labels_metal1=cap_utils.load_data('./Cases/STACK3L/SUB-metal1-metal2_STACK3L/SUB-metal1-metal2_STACK3L.tbl.text', 0, 144)
labels_metal2=cap_utils.load_data('./Cases/STACK3L/SUB-metal2-metal3_STACK3L/SUB-metal2-metal3_STACK3L.tbl.text', 0, 144)
labels_metal3=cap_utils.load_data('./Cases/STACK3L/SUB-metal1-metal3_STACK3L/SUB-metal1-metal3_STACK3L.tbl.text', 0, 144)
labels_metal4=cap_utils.load_data('./Cases/STACK3L/STACK3L_single_expand/1/final_output.txt', 0, 864)[:,[1,2,3,5,6,7,9]]
labels_metal5=cap_utils.load_data('./Cases/STACK3L/STACK3L_single_expand/2/final_output.txt', 0, 864)[:,[1,2,3,5,6,7,9]]
labels=torch.cat((labels_metal1, labels_metal2, labels_metal3,))
labels_train = labels[train_index, :].cuda()
# labels_test = labels[test_index, :]
labels_test = labels_metal1

inputs_metal1=torch.tensor(parser_single_expand.parser(pattern='STACK3L', metal='12', pattern_path='./Cases',gen=False, input_num=0),dtype=torch.float32)
inputs_metal2=torch.tensor(parser_single_expand.parser(pattern='STACK3L', metal='23', pattern_path='./Cases',gen=False, input_num=0),dtype=torch.float32)
inputs_metal3=torch.tensor(parser_single_expand.parser(pattern='STACK3L', metal='13', pattern_path='./Cases',gen=False, input_num=0),dtype=torch.float32)
inputs_metal4=torch.tensor(parser_single_expand.parser(pattern='STACK3L', metal='11', pattern_path='./Cases/STACK3L/STACK3L_single_expand/1/input_files', gen=True, input_num=1728),dtype=torch.float32)
inputs_metal5=torch.tensor(parser_single_expand.parser(pattern='STACK3L', metal='11', pattern_path='./Cases/STACK3L/STACK3L_single_expand/2/intput_files', gen=True, input_num=1728),dtype=torch.float32)

inputs=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3))
inputs_train=inputs[train_index, :].cuda()
# inputs_test=inputs[test_index, :]
inputs_test=inputs_metal1

file = open("./log/STACK3L/train_index.txt", "w+")
file.write(str(train_index))
file.close()
file = open("./log/STACK3L/val_index.txt", "w+")
file.write(str(test_index))
file.close()


"""训练、测试模型"""

# 初始化模型、损失函数和优化器
batch_size=2048
model = CpMLP.CpMLP(inchans=42, hidden1=4096, hidden2=1024, hidden3=1024, hidden4=128, hidden5=128, outchans=7)
# for param in model.parameters():
#     torch.nn.init.normal_(param, mean=0, std=0.01)
state_dict = torch.load('./model/STACK3L/saved/model_STACK3L_CpMLP.pt')
model.load_state_dict(state_dict)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
""" for epoch in range(10000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train)
    if train_loss<0.01:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}") """

# 验证并记录
model.eval().to('cpu')
predict=model(inputs_test)
weights=torch.abs(labels_test)/(torch.abs(labels_test).sum(dim=1, keepdim=True))
aveloss=torch.abs(model(inputs_test)-labels_test)/torch.max(torch.abs(labels_test), torch.tensor(0.00001))*weights
test_error=cap_utils.test_model(model, inputs_test, labels_test)
train_error=cap_utils.test_model(model, inputs_train.cpu(), labels_train.cpu())

print(f"Total Test Loss: {test_error}")
print(f"Total Test Loss: {test_error.mean()}")
file = open("./log/STACK3L/results.txt", "w+")
file.write(f"predict\n {predict.detach().numpy()}\n Test Loss for every cap\n {aveloss.detach().numpy()}\n Test Loss\n {test_error.detach().numpy()}\n Average Test Loss\n {test_error.mean()}\n Train Loss\n {(train_error.detach().numpy())}")
file.close()

# 保存模型
torch.save(model.state_dict(), './model/STACK3L/model_STACK3L_CpMLP.pt')