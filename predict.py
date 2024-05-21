from model.PLATE2L import CpMLP as model_PLATE2L
from model.PLATE3L import CpMLP as model_PLATE3L
from model.STACK3L import CpMLP as model_STACK3L
from pathlib import Path
import torch
from utils import parser_final
import argparse
import os

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device= 'cpu'


def predict_PLATE2L(input, script_dir):
    model = model_PLATE2L.CpMLP(inchans=26, hidden1=4096, hidden2=1024, hidden3=512, hidden4=128, hidden5=128, outchans=3)
    state_dict = torch.load(script_dir/'model/PLATE2L/saved/model_PLATE2L_CpMLP.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def predict_STACK3L(input, script_dir):
    model = model_STACK3L.CpMLP(inchans=42, hidden1=4096, hidden2=1024, hidden3=1024, hidden4=128, hidden5=128, outchans=7)
    state_dict = torch.load(script_dir/'model/STACK3L/saved/model_STACK3L_CpMLP.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def predict_PLATE3L(input, script_dir, subtype):
    if subtype==0:
        return predict_PLATE3L_c1(input, script_dir)
    elif subtype==1:
        return predict_PLATE3L_c1c2(input, script_dir)
    elif subtype==2:
        return predict_PLATE3L_c1lEnv(input, script_dir)
    elif subtype==3:
        return predict_PLATE3L_c1c2lEnvrEnv(input, script_dir)


def predict_PLATE3L_c1(input, script_dir):
    model = model_PLATE3L.CpMLP(inchans=15, hidden1=2048, hidden2=1024, hidden3=1024, hidden4=512, hidden5=128, outchans=3)
    state_dict = torch.load(script_dir/'model/PLATE3L/saved/model_PLATE3L_c1.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def predict_PLATE3L_c1c2(input, script_dir):
    model = model_PLATE3L.CpMLP(inchans=21, hidden1=2048, hidden2=1024, hidden3=512, hidden4=512, hidden5=128, outchans=4)
    state_dict = torch.load(script_dir/'model/PLATE3L/saved/model_PLATE3L_c1c2.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def predict_PLATE3L_c1lEnv(input, script_dir):
    model = model_PLATE3L.CpMLP(inchans=20, hidden1=3072, hidden2=1024, hidden3=512, hidden4=512, hidden5=256, outchans=3)
    state_dict = torch.load(script_dir/'model/PLATE3L/saved/model_PLATE3L_c1lEnv.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def predict_PLATE3L_c1c2lEnvrEnv(input, script_dir):
    model = model_PLATE3L.CpMLP(inchans=31, hidden1=3072, hidden2=1024, hidden3=1024, hidden4=512, hidden5=128, outchans=5)
    state_dict = torch.load(script_dir/'model/PLATE3L/saved/model_PLATE3L_c1c2lEnvrEnv.pt', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    output = model(input)
    return output

def write_output_single(output, out_dir, type, subtype=0):
    if type == 'PLATE2L':
        file = open(out_dir+'/results.txt', "w+")
        file.write(f"c1: {output[0][0].item()}\n")
        file.write(f"c1t: 0.00000000\n")
        file.write(f"c1b: {output[0][1].item()}\n")
        file.write(f"c1tb: 0.00000000\n")
        file.write(f"c1e: {output[0][2].item()}\n")
        file.close()
    elif type=='STACK3L':
        file = open(out_dir+'/results_1.txt', "w+")
        file.write(f"c12: {output[0][0].item()}\n")
        file.write(f"c13: {output[0][1].item()}\n")
        file.write(f"c14: {output[0][2].item()}\n")
        file.write(f"c1t: 0.00000000\n")
        file.write(f"c1b: {output[0][3].item()}\n")
        file.close()
        file = open(out_dir+'/results_2.txt', "w+")
        file.write(f"c23: {output[0][4].item()}\n")
        file.write(f"c24: {output[0][5].item()}\n")
        file.write(f"c2t: 0.00000000\n")
        file.write(f"c2b: {output[0][6].item()}\n")
        file.close()
    elif type=='PLATE3L':
        if subtype==0:
            file = open(out_dir+'/results_1.txt', "w+")
            file.write(f"c12: 0.00000000\n")
            file.write(f"c1t: {output[0][0].item()}\n")
            file.write(f"c1b: {output[0][1].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_2.txt', "w+")
            file.write(f"c1tb: {output[0][2].item()}\n")
            file.close()
        elif subtype==1:
            file = open(out_dir+'/results_1.txt', "w+")
            file.write(f"c12: 0.00000000\n")
            file.write(f"c1t: {output[0][1].item()}\n")
            file.write(f"c1b: {output[0][2].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_2.txt', "w+")
            file.write(f"c1tb: {output[0][3].item()}\n")
            file.close()
        elif subtype==2:
            file = open(out_dir+'/results_1.txt', "w+")
            file.write(f"c12: {output[0][0].item()}\n")
            file.write(f"c1t: {output[0][1].item()}\n")
            file.write(f"c1b: {output[0][2].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_2.txt', "w+")
            file.write(f"c1tb: {output[0][3].item()}\n")
            file.close()
        elif subtype==3:
            file = open(out_dir+'/results_1.txt', "w+")
            file.write(f"c12: {output[0][0].item()}\n")
            file.write(f"c1t: {output[0][1].item()}\n")
            file.write(f"c1b: {output[0][2].item()}\n")
            file.write(f"c1e: {output[0][4].item()}\n")
            file.close()
            file = open(out_dir+'/results_2.txt', "w+")
            file.write(f"c1tb: {output[0][3].item()}\n")
            file.close()
    

def write_output_not_single(output, out_dir, num, type, subtype=''):
    if type == 'PLATE2L':
        for i in range(num):
            file = open(out_dir+'/results_input'+str(i+1)+'.txt', "w+")
            file.write(f"c1: {output[i][0].item()}\n")
            file.write(f"c1t: 0.00000000\n")
            file.write(f"c1b: {output[i][1].item()}\n")
            file.write(f"c1tb: 0.00000000\n")
            file.write(f"c1e: {output[i][2].item()}\n")
            file.close()
    elif type=='STACK3L':
        for i in range(num//2):
            file = open(out_dir+'/results_input'+str(2*i+1)+'.txt', "w+")
            file.write(f"c12: {output[i][0].item()}\n")
            file.write(f"c13: {output[i][1].item()}\n")
            file.write(f"c14: {output[i][2].item()}\n")
            file.write(f"c1t: 0.00000000\n")
            file.write(f"c1b: {output[i][3].item()}\n")
            file.close()
            file = open(out_dir+'/results_input'+str(2*i+2)+'.txt', "w+")
            file.write(f"c23: {output[i][4].item()}\n")
            file.write(f"c24: {output[i][5].item()}\n")
            file.write(f"c2t: 0.00000000\n")
            file.write(f"c2b: {output[i][6].item()}\n")
            file.close()
    elif type=='PLATE3L':
        if subtype==0:
            file = open(out_dir+'/results_input'+str(num*2+1)+'.txt', "w+")
            file.write(f"c12: 0.00000000\n")
            file.write(f"c1t: {output[0].item()}\n")
            file.write(f"c1b: {output[1].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_input'+str(num*2+2)+'.txt', "w+")
            file.write(f"c1tb: {output[2].item()}\n")
            file.close()
        elif subtype==2:
            file = open(out_dir+'/results_input'+str(num*2+1)+'.txt', "w+")
            file.write(f"c12: 0.00000000\n")
            file.write(f"c1t: {output[0].item()}\n")
            file.write(f"c1b: {output[1].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_input'+str(num*2+2)+'.txt', "w+")
            file.write(f"c1tb: {output[2].item()}\n")
            file.close()
        elif subtype==1:
            file = open(out_dir+'/results_input'+str(num*2+1)+'.txt', "w+")
            file.write(f"c12: {output[0].item()}\n")
            file.write(f"c1t: {output[1].item()}\n")
            file.write(f"c1b: {output[2].item()}\n")
            file.write(f"c1e: 0.00000000\n")
            file.close()
            file = open(out_dir+'/results_input'+str(num*2+2)+'.txt', "w+")
            file.write(f"c1tb: {output[3].item()}\n")
            file.close()
        elif subtype==3:
            file = open(out_dir+'/results_input'+str(num*2+1)+'.txt', "w+")
            file.write(f"c12: {output[0].item()}\n")
            file.write(f"c1t: {output[1].item()}\n")
            file.write(f"c1b: {output[2].item()}\n")
            file.write(f"c1e: {output[4].item()}\n")
            file.close()
            file = open(out_dir+'/results_input'+str(num*2+2)+'.txt', "w+")
            file.write(f"c1tb: {output[3].item()}\n")
            file.close()


def main(args):
    script_dir=Path.cwd()
    # torch.set_printoptions(precision=8)
    feature_matrix, model_index, pattern = parser_final.parser(
        single=args.single,
        file1=args.file1_path,
        file2=args.file2_path,
        pre=args.pre,
        post=args.post,
        input_num=args.input_num,
        directory=args.directory,
    )
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.single == True:
        if pattern == 'PLATE2L':
            input=torch.tensor(feature_matrix,dtype=torch.float32)
            output = predict_PLATE2L(input, script_dir)
            write_output_single(output, args.out_dir, "PLATE2L")
        elif pattern == 'STACK3L':
            input=torch.tensor(feature_matrix,dtype=torch.float32)
            output = predict_STACK3L(input, script_dir)
            write_output_single(output, args.out_dir, "STACK3L")
        elif pattern == 'PLATE3L':
            for i in range(len(model_index)):
                if len(model_index[i])==1:
                    model_type = i
            input=torch.tensor(feature_matrix[model_type],dtype=torch.float32)
            if model_type == 0:
                output = predict_PLATE3L_c1(input, script_dir)
            elif model_type == 1:
                output = predict_PLATE3L_c1c2(input, script_dir)
            elif model_type == 2:
                output = predict_PLATE3L_c1lEnv(input, script_dir)
            elif model_type == 3:
                output = predict_PLATE3L_c1c2lEnvrEnv(input, script_dir)
            write_output_single(output, args.out_dir, "PLATE3L", subtype=model_type)
    elif args.single == False:
        if pattern == 'PLATE2L':
            input=torch.tensor(feature_matrix,dtype=torch.float32)
            output = predict_PLATE2L(input, script_dir)
            write_output_not_single(output, args.out_dir, args.input_num, "PLATE2L")
        elif pattern == 'STACK3L':
            input=torch.tensor(feature_matrix,dtype=torch.float32)
            output = predict_STACK3L(input, script_dir)
            # print(output.shape)
            write_output_not_single(output, args.out_dir, args.input_num, "STACK3L")
        elif pattern == 'PLATE3L':
            for subtype in range(4):
                # print(subtype)
                input=torch.tensor(feature_matrix[subtype],dtype=torch.float32)
                output = predict_PLATE3L(input, script_dir, subtype)
                # print(output.shape[0])
                for i in range(output.shape[0]):
                    # print(model_index[subtype][i])
                    write_output_not_single(output[i], args.out_dir, type="PLATE3L", num=model_index[subtype][i], subtype=subtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1_path", type=str, required=False)
    parser.add_argument("--file2_path", type=str, required=False)
    parser.add_argument("--pre", type=str, required=False)
    parser.add_argument("--post", type=str, required=False)
    parser.add_argument("--input_num", type=int, required=False)
    parser.add_argument("--directory", type=str, required=False)
    parser.add_argument("--single", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)