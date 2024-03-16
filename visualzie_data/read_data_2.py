import numpy as np
import torch

class Master:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points

    def print_info(self):
        print("master's name: ", self.name)
        print("number of this master: ", self.num)
        print("points: ", self.points)

    def Array(self):
        array=np.zeros(4)
        array[0]=self.points[0][0]
        array[1]=self.points[0][1]
        array[2]=self.points[1][0]-self.points[0][0]
        array[3]=self.points[3][1]-self.points[0][1]
        return array

class Env:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points

    def print_info(self):
        print("environmental conductor's name: ", self.name)
        print("number of this environmental conductor: ", self.num)
        print("points: ", self.points)

    def Array(self):
        array=np.zeros(4)
        array[0]=self.points[0][0]
        array[1]=self.points[0][1]
        array[2]=self.points[1][0]-self.points[0][0]
        array[3]=self.points[3][1]-self.points[0][1]
        return array


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            if len(lines) >= 1:
                # master
                master_name = lines[0].strip()
                num_of_mp = int(lines[1].strip())
                k = 2
                master_points = []
                for _ in range(num_of_mp):
                    master_points.append([float(num)
                                         for num in lines[k].split()])
                    k = k+1
                master = Master(master_name, num_of_mp, points=master_points)

                k = k+1
                num_of_env = int(lines[k].strip())
                k = k+1
                env_list = []
                for _ in range(num_of_env):
                    num_of_env_n = int(lines[k+1].strip())
                    env_points = []
                    for j1 in range(num_of_env_n):
                        env_points.append([float(num)
                                           for num in lines[k+2+j1].split()])
                    env_instance = Env(
                        name=lines[k].strip(), num=num_of_env_n,
                        points=env_points)
                    env_list.append(env_instance)
                    k = k+1+num_of_env_n+2

                """ k = k+1
                num_of_dielectric = int(lines[k].strip())
                k = k+1
                dielectric_list = []
                for _ in range(num_of_dielectric):
                    num_of_dielectric_n = int(lines[k+1].strip())
                    dielectric_points = []
                    for j2 in range(num_of_dielectric_n):
                        dielectric_points.append(
                            [float(num) for num in lines[k+2+j2].split()])
                    dielectric_instance = Dielectric(
                        name=lines[k].strip(), num=num_of_dielectric_n,
                        points=dielectric_points,
                        er=int(lines[k+num_of_dielectric_n+2].strip()))
                    dielectric_list.append(dielectric_instance)
                    k = k+num_of_dielectric_n+4

                k = k+1
                num_of_boundary_polygon_points = int(lines[k].strip())
                k = k+1
                boundpoly_points = []
                for _ in range(num_of_boundary_polygon_points):
                    boundpoly_points.append([float(num)
                                            for num in lines[k].split()])
                    k = k+1
                boundpoly = Boundary_Polygon(
                    num_of_boundary_polygon_points, boundpoly_points) """

                # 返回结果，可以根据需要修改返回值
                return master, env_list #, dielectric_list, boundpoly
            else:
                print("文件行数不足，无法获取必要信息。")
                return None
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def load_data(input_num=64):
    file_path = './data/type1_data/BEM_INPUT_1_43652.txt'
    master, env_list= read_txt_file(file_path)
    array=master.Array()
    for env in env_list:
            array_i = env.Array()
            array = np.vstack((array, array_i))
    array=torch.from_numpy(array)
    array = array.unsqueeze(0)      # 增加一个维度
    for i in range(1, input_num):
        file_path = './data/type1_data/BEM_INPUT_'+str(i+1)+'_43652.txt'
        master, env_list= read_txt_file(file_path)
        array_i=master.Array()
        for env in env_list:
            array_j = env.Array()
            array_i = np.vstack((array_i, array_j))
        array_i=torch.from_numpy(array_i)
        array_i = array_i.unsqueeze(0)
        array=torch.cat((array, array_i),dim=0)
    return array.permute(0, 2, 1)                    # B, C, N

print(load_data())