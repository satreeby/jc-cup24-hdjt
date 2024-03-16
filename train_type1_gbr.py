from parser_work import parser_6_features
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# 评价标准
def weighted_average_loss(outputs, labels):
    '''加权平均误差, [batch_size, 1] 的矩阵'''
    weights=abs(labels)/(abs(labels).sum(axis=1, keepdims=True))
    aveloss=abs(outputs-labels)/abs(labels)*weights
    return aveloss.sum(axis=1)

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
    dataset_labels=np.array(read_data(labels_path),dtype=float)
    labels=dataset_labels[start: end]
    return labels

labels_train = load_data('C:\project\python\HuadaJiutian\labels\labels_type1.txt', 0, 50)
labels_test = load_data('C:\project\python\HuadaJiutian\labels\labels_type1.txt', 50, 64)


# 获取输入数据

inputs=np.array(parser_6_features.parser(type=1, Fpath='./data'),dtype=float)
inputs_train=inputs[0:50, :]
inputs_test=inputs[50:, :]


# 创建GradientBoostingRegressor模型
base_model_gbr = GradientBoostingRegressor(learning_rate=0.8, subsample=1, n_estimators=100)

# 使用MultiOutputRegressor包装模型
model_gbr = MultiOutputRegressor(base_model_gbr)
model_gbr.fit(inputs_train, labels_train)


# 对测试集进行预测
pred = model_gbr.predict(inputs_test)

# 评估
print(f"GBR Total test loss: {weighted_average_loss(pred, labels_test)}")