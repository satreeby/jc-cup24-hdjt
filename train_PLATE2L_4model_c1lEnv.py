import torch
import numpy as np
from torch.utils import data
from model.PLATE2L import CpMLP
from parser_work import parser_4model_for1
from utils import cap_utils
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda:0')

# 加载训练，验证数据集以及标签

feature_matrix_metal1, model_index_metal1 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='1', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal1=torch.tensor(feature_matrix_metal1[2],dtype=torch.float32)
feature_matrix_metal2, model_index_metal2 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='2', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal2=torch.tensor(feature_matrix_metal2[2],dtype=torch.float32)
feature_matrix_metal3, model_index_metal3 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='3', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal3=torch.tensor(feature_matrix_metal3[2],dtype=torch.float32)
feature_matrix_extentions, _ = parser_4model_for1.parser_features(pattern='PLATE2L', metal='1', 
                                         pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1lEnv/PLATE2L_random_file',
                                         generation=True, input_num=3630,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal_extentions=torch.tensor(feature_matrix_extentions[2],dtype=torch.float32)

inputs_train = inputs_metal_extentions
min_value = inputs_train.min(dim=0)[0]
max_value = inputs_train.max(dim=0)[0]
# print(min_value)
# print(max_value)
inputs_train = (inputs_train - min_value) / (max_value - min_value) * 2 - 1
inputs_train = inputs_train.cuda()
inputs_val=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3))
inputs_val = (inputs_val - min_value) / (max_value - min_value) * 2 - 1

print(inputs_metal1.shape)
print(inputs_metal2.shape)
print(inputs_metal3.shape)
print(inputs_metal_extentions.shape)


labels_metal1=cap_utils.load_data('./Cases/PLATE2L/SUB-metal1_PLATE2L/SUB-metal1_PLATE2L.tbl.text', 0, 324)
labels_metal2=cap_utils.load_data('./Cases/PLATE2L/SUB-metal2_PLATE2L/SUB-metal2_PLATE2L.tbl.text', 0, 324)
labels_metal3=cap_utils.load_data('./Cases/PLATE2L/SUB-metal3_PLATE2L/SUB-metal3_PLATE2L.tbl.text', 0, 324)
labels_metal_extentions=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1lEnv/final_output.txt', 0, 3630)
labels_metal1=labels_metal1[model_index_metal1[2],1]
labels_metal2=labels_metal2[model_index_metal2[2],1]
labels_metal3=labels_metal3[model_index_metal3[2],1]

labels_train=labels_metal_extentions.cuda()
labels_val=torch.cat((labels_metal1, labels_metal2, labels_metal3)).unsqueeze(dim=1)

print(labels_train.shape)
print(labels_val.shape)

# 初始化模型、损失函数和优化器
batch_size=4096
model = CpMLP.CpMLP(inchans=15, hidden1=2048, hidden2=512, hidden3=256, hidden4=128, hidden5=128, outchans=1)
# for param in model.parameters():
#     torch.nn.init.normal_(param, mean=0, std=0.01)
state_dict = torch.load('./model/PLATE2L/saved/model_PLATE2L_4model_c1lEnv.pt')
model.load_state_dict(state_dict)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 ,weight_decay=0.)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)


""" for epoch in range(3000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
    if train_loss<0.009:
        break

# 保存模型
torch.save(model.state_dict(), './model/PLATE2L/model_PLATE2L_4model_c1lEnv.pt') """

# 验证并记录
model.eval().to('cpu')
predict=model(inputs_val)
weights=torch.abs(labels_val)/(torch.abs(labels_val).sum(dim=1, keepdim=True))
aveloss=torch.abs(model(inputs_val)-labels_val)/torch.max(torch.abs(labels_val), torch.tensor(0.00001))*weights
test_error=cap_utils.test_model(model, inputs_val, labels_val, device='cpu')
train_error=cap_utils.test_model(model, inputs_train.cpu(), labels_train.cpu(), device='cpu')

print(f"Total Test Loss: {test_error}")
print(f"Total Test Loss: {test_error.mean()}")
file = open("./log/PLATE2L/results_ext.txt", "w+")
file.write(f"predict\n {predict.detach().numpy()}\n Test Loss for every cap\n {aveloss.detach().numpy()}\n Test Loss\n {test_error.detach().numpy()}\n Average Test Loss\n {test_error.mean()}\n Train Loss\n {(train_error.detach().numpy())}")
file.close()

