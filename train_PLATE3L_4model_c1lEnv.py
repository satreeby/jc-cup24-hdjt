import torch
import numpy as np
from torch.utils import data
from model.PLATE3L import CpMLP
from parser_work import parser_4model_for2
from utils import cap_utils
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)

device = torch.device('cuda:0')

# 加载训练，验证数据集以及标签

feature_matrix_metal1, model_index_metal1 = parser_4model_for2.parser_features(pattern='PLATE3L', metal='12', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal1=torch.tensor(feature_matrix_metal1[2],dtype=torch.float32)
feature_matrix_metal2, model_index_metal2 = parser_4model_for2.parser_features(pattern='PLATE3L', metal='23', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal2=torch.tensor(feature_matrix_metal2[2],dtype=torch.float32)
feature_matrix_metal3, model_index_metal3 = parser_4model_for2.parser_features(pattern='PLATE3L', metal='13', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal3=torch.tensor(feature_matrix_metal3[2],dtype=torch.float32)
feature_matrix_metal4, model_index_metal4= parser_4model_for2.parser_features(pattern='PLATE3L', metal='1', 
                                         pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/1/input_files',
                                         generation=True, input_num=1944,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal4=torch.tensor(feature_matrix_metal4[2],dtype=torch.float32)
feature_matrix_metal5, model_index_metal5= parser_4model_for2.parser_features(pattern='PLATE3L', metal='1', 
                                         pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/2/input_files',
                                         generation=True, input_num=1944,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal5=torch.tensor(feature_matrix_metal5[2],dtype=torch.float32)
feature_matrix_metal6, model_index_metal6= parser_4model_for2.parser_features(pattern='PLATE3L', metal='1', 
                                         pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/3/input_files',
                                         generation=True, input_num=1944,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal6=torch.tensor(feature_matrix_metal6[2],dtype=torch.float32)
feature_matrix_metal7, model_index_metal7= parser_4model_for2.parser_features(pattern='PLATE3L', metal='1', 
                                         pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/4/input_files',
                                         generation=True, input_num=1944,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
inputs_metal7=torch.tensor(feature_matrix_metal7[2],dtype=torch.float32)


# 划分训练集和验证集
n_total = 270
n_train = 216
train_index = np.array(cap_utils.getRandomIndex(n_total=n_total, n_sample=n_train))
test_index = np.delete(np.arange(n_total), train_index)

# labels
labels_metal1=cap_utils.load_data('./Cases/PLATE3L/SUB-metal1-metal2_PLATE3L/SUB-metal1-metal2_PLATE3L.tbl.text', 0, 324)[model_index_metal1[2],:][:,[1,2,3]]
labels_metal2=cap_utils.load_data('./Cases/PLATE3L/SUB-metal2-metal3_PLATE3L/SUB-metal2-metal3_PLATE3L.tbl.text', 0, 324)[model_index_metal2[2],:][:,[1,2,3]]
labels_metal3=cap_utils.load_data('./Cases/PLATE3L/SUB-metal1-metal3_PLATE3L/SUB-metal1-metal3_PLATE3L.tbl.text', 0, 324)[model_index_metal3[2],:][:,[1,2,3]]
labels_metal4=cap_utils.load_data('./Cases/PLATE3L/PLATE3L_single_expand/1/final_output.txt', 0, 972)[model_index_metal4[2],:][:,[2, 3, 4]]
labels_metal5=cap_utils.load_data('./Cases/PLATE3L/PLATE3L_single_expand/2/final_output.txt', 0, 972)[model_index_metal5[2],:][:,[2, 3, 4]]
labels_metal6=cap_utils.load_data('./Cases/PLATE3L/PLATE3L_single_expand/3/final_output.txt', 0, 972)[model_index_metal6[2],:][:,[2, 3, 4]]
labels_metal7=cap_utils.load_data('./Cases/PLATE3L/PLATE3L_single_expand/4/final_output.txt', 0, 972)[model_index_metal7[2],:][:,[2, 3, 4]]
labels=torch.cat((labels_metal1, labels_metal2, labels_metal3, labels_metal4, labels_metal5, labels_metal6, labels_metal7))
labels_train = labels[train_index, :].cuda()
labels_test = labels[test_index, :]

print(labels.shape)
# inputs
#inputs_metal1=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='12', pattern_path='./Cases', gen=False, input_num=0),dtype=torch.float32)
#inputs_metal2=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='23', pattern_path='./Cases', gen=False, input_num=0),dtype=torch.float32)
#inputs_metal3=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='13', pattern_path='./Cases', gen=False, input_num=0),dtype=torch.float32)
#inputs_metal4=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='13', pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/1/input_files', gen=True, input_num=1944),dtype=torch.float32)
#inputs_metal5=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='13', pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/2/input_files', gen=True, input_num=1944),dtype=torch.float32)
#inputs_metal6=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='13', pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/3/input_files', gen=True, input_num=1944),dtype=torch.float32)
#inputs_metal7=torch.tensor(parser_single_expand3.parser(pattern='PLATE3L', metal='13', pattern_path='./Cases/PLATE3L/PLATE3L_single_expand/4/input_files', gen=True, input_num=1944),dtype=torch.float32)
inputs=torch.cat((inputs_metal1, inputs_metal2, inputs_metal3, inputs_metal4, inputs_metal5, inputs_metal6, inputs_metal7))
inputs_train=inputs[train_index, :].cuda()
inputs_test=inputs[test_index, :]
print(inputs.shape)
# loggers
file = open("./log/PLATE3L/train_index.txt", "w+")
file.write(str(train_index))
file.close()
file = open("./log/PLATE3L/val_index.txt", "w+")
file.write(str(test_index))
file.close()


# 训练、测试模型

# 初始化模型、损失函数和优化器

batch_size=256
model = CpMLP.CpMLP(inchans=21, hidden1=3072, hidden2=1024, hidden3=512, hidden4=512, hidden5=256, outchans=3)
for param in model.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-7)
train_loader = data.DataLoader(data.TensorDataset(inputs_train, labels_train), batch_size, shuffle=True)

# 训练
for epoch in range(20000):
    train_loss = cap_utils.train_model(model, criterion, optimizer, train_loader, inputs_train, labels_train)
    # with torch.no_grad():
    #     loss_i = criterion(model(inputs_train), labels_train).mean()
    if train_loss<0.01:
        break
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

# 保存模型
torch.save(model.state_dict(), './model/PLATE3L/model_PLATE3L_c1lEnv.pt')

# 验证并记录
model.eval().to('cpu')
predict=model(inputs_test)
weights=torch.abs(labels_test)/(torch.abs(labels_test).sum(dim=1, keepdim=True))
aveloss=torch.abs(model(inputs_test)-labels_test)/torch.max(torch.abs(labels_test), torch.tensor(0.00001))*weights
test_error=cap_utils.test_model(model, inputs_test, labels_test)
train_error=cap_utils.test_model(model, inputs_train.cpu(), labels_train.cpu())

print(f"Total Test Loss: {test_error}")
print(f"Total Test Loss: {test_error.mean()}")
file = open("./log/PLATE3L/results.txt", "w+")
file.write(f"predict\n {predict.detach().numpy()}\n Test Loss for every cap\n {aveloss.detach().numpy()}\n Test Loss\n {test_error.detach().numpy()}\n Average Test Loss\n {test_error.mean()}\n Train Loss\n {(train_error.detach().numpy())}")
file.close()