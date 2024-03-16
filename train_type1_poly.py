import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import parser_1

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

labels_train = load_data('C:\project\python\HuadaJiutian\labels_type1.txt', 0, 60)
labels_test = load_data('C:\project\python\HuadaJiutian\labels_type1.txt', 60, 64)


# 获取输入数据

inputs=np.array(parser_1.parser(type=1, Fpath='./data'),dtype=float)
inputs_train=inputs[0:60, :]
inputs_test=inputs[60:, :]


# 使用多项式特征扩展
for i in range(15):
    poly = PolynomialFeatures(degree=i)
    X_poly = poly.fit_transform(inputs_train)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, labels_train)

    # 打印模型系数
    # print("模型系数：", model_poly.coef_)

    # 使用训练好的模型进行预测
    y_pred = model_poly.predict(poly.fit_transform(inputs_test))

    # 打印预测结果
    # print("预测结果：", y_pred)

    # 打印误差
    print(f"degree: {i}, Total test loss: {weighted_average_loss(y_pred, labels_test)}")
