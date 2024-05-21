import torch
import numpy as np
from torch.utils import data
from model.PLATE2L import CpMLP
from parser_work import parser_4model_for1
from utils import cap_utils
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

device = torch.device('cuda:0')

n_total = 1188
n_train = 950
train_index = np.array(cap_utils.getRandomIndex(n_total=n_total, n_sample=n_train))
test_index = np.delete(np.arange(n_total), train_index)


# 加载训练，验证数据集以及标签

feature_matrix_metal1, model_index_metal1 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='1', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal1=torch.tensor(feature_matrix_metal1[3],dtype=torch.float32)
feature_matrix_metal2, model_index_metal2 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='2', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal2=torch.tensor(feature_matrix_metal2[3],dtype=torch.float32)
feature_matrix_metal3, model_index_metal3 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='3', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal3=torch.tensor(feature_matrix_metal3[3],dtype=torch.float32)
feature_matrix_metal456, model_index_metal456 = parser_4model_for1.parser_features(pattern='PLATE2L', metal='3', 
                                         pattern_path='./Cases/PLATE2L/SUB-metal456_PLATE2L/input',
                                         generation=True, input_num=972,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal456=torch.tensor(feature_matrix_metal456[3],dtype=torch.float32)
feature_matrix_extentions, _ = parser_4model_for1.parser_features(pattern='PLATE2L', metal='1', 
                                         pattern_path='./Cases/PLATE2L/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_random_file',
                                         generation=True, input_num=46170,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal_extentions=torch.tensor(feature_matrix_extentions[3],dtype=torch.float32)

inputs = torch.cat((inputs_metal1, inputs_metal2, inputs_metal3, inputs_metal456))
inputs_train = inputs[train_index, :]
min_value = inputs_train.min(dim=0)[0]
max_value = inputs_train.max(dim=0)[0]
print(min_value)
print(max_value)
inputs_train = (inputs_train - min_value) / (max_value - min_value) * 2 - 1
inputs_train = inputs_train.cuda()
inputs_val = inputs[test_index, :]
inputs_val = (inputs_val - min_value) / (max_value - min_value) * 2 - 1

print(inputs_train.shape)
print(inputs_val.shape)



labels_metal1=cap_utils.load_data('./Cases/PLATE2L/SUB-metal1_PLATE2L/SUB-metal1_PLATE2L.tbl.text', 0, 324)
labels_metal2=cap_utils.load_data('./Cases/PLATE2L/SUB-metal2_PLATE2L/SUB-metal2_PLATE2L.tbl.text', 0, 324)
labels_metal3=cap_utils.load_data('./Cases/PLATE2L/SUB-metal3_PLATE2L/SUB-metal3_PLATE2L.tbl.text', 0, 324)
labels_metal456=cap_utils.load_data('./Cases/PLATE2L/SUB-metal456_PLATE2L/SUB-metal456_PLATE2L.tbl.text', 0, 972)
labels_metal_extentions=cap_utils.load_data('./Cases/PLATE2L/PLATE2L_4model_c1c2lEnvrEnv/PLATE2L_4model_c1c2lEnvrEnv/final_output.txt', 0, 46170)
labels_metal1=labels_metal1[model_index_metal1[3],:]
labels_metal2=labels_metal2[model_index_metal2[3],:]
labels_metal3=labels_metal3[model_index_metal3[3],:]
labels_metal456=labels_metal456[model_index_metal456[3],:]

# labels_train=labels_metal_extentions[:,[1, 3, 5]].cuda()
labels=torch.cat((labels_metal1, labels_metal2, labels_metal3, labels_metal456))
labels_train = labels[train_index,:].cuda()
labels_val = labels[test_index,:]

print(labels_train.shape)
print(labels_val.shape)

# 初始化模型、损失函数和优化器
batch_size=4096
model = CpMLP.CpMLP(inchans=27, hidden1=4096, hidden2=1024, hidden3=1024, hidden4=256, hidden5=256, outchans=3)
# for param in model.parameters():
#     torch.nn.init.normal_(param, mean=0, std=0.01)
state_dict = torch.load('./model/PLATE2L/saved/model_PLATE2L_4model_c1c2lEnvrEnv.pt')
model.load_state_dict(state_dict)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 ,weight_decay=0.)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)


for epoch in range(1000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train, device)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
    if train_loss<0.009:
        break

# 保存模型
torch.save(model.state_dict(), './model/PLATE2L/model_PLATE2L_4model_c1c2lEnvrEnv.pt')

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

