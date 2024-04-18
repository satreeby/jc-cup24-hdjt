import torch
from parser_work import parser
from model.PLATE2L import CpMLP
import time


inputs_metal1=torch.tensor(parser.parser(pattern='PLATE2L', metal='1', pattern_path='./Cases'),dtype=torch.float32)
model=CpMLP.CpMLP(inchans=14, hidden1=500, hidden2=300, hidden3=200, hidden4=200, outchans=3)
state_dict = torch.load('./model/PLATE2L/model_PLATE2L_CpMLP.pt')
model.load_state_dict(state_dict)

time_start = time.time()            # 记录开始时间
model(inputs_metal1)
time_end = time.time()              # 记录结束时间

time_sum = time_end - time_start    # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
