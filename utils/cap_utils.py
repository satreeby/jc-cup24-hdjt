import torch
import numpy as np


def weighted_average_loss(outputs, labels):
    '''评价标准，加权平均误差, [batch_size, 1] 的矩阵'''
    weights=torch.abs(labels)/(torch.abs(labels).sum(dim=1, keepdim=True))
    aveloss=torch.abs(outputs-labels)/ torch.max(torch.abs(labels), torch.tensor(0.0000001))*weights
    return aveloss.sum(dim=1)


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


def getRandomIndex(n_total, n_sample):
	# 索引范围为[0, total)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(np.arange(n_total), size=n_sample, replace=False)
    return index


def train_model(model, criterion, optimizer, dataloader, features_train, labels_train, device='cuda:0'):
    model.to(device)
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    average_loss = weighted_average_loss(model(features_train), labels_train).max()
    return average_loss

def test_model(model, features_test, labels_test, device='cpu'):
    model=model.to(device)
    model.eval()
    features_test=features_test
    labels_test=labels_test
    test_loss= weighted_average_loss(model(features_test), labels_test)
    return test_loss