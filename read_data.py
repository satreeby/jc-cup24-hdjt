import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

FACTOR = 100

fig, ax = plt.subplots()
ax.set_xlim([-0.75, 1.3])
ax.set_ylim([-0.006, 0.3])

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
        array=np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0]=self.points[i][0]
            array[i][1]=self.points[i][1]
        return array

    def List_for_Vision(self):
        x = []
        y = []
        for i in range(self.num):
            x.append(self.points[i][0])
            y.append(self.points[i][1])
        x.append(self.points[0][0])
        y.append(self.points[0][1])
        return x, y
    
    def Tensorize(self):
        tensor=torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0]=self.points[i][0]
            tensor[i][1]=self.points[i][1]
        return tensor
    
    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)
        ax.text(midX, midY, self.name)

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
        array=np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0]=self.points[i][0]
            array[i][1]=self.points[i][1]
        return array

    def List_for_Vision(self):
        x = []
        y = []
        for i in range(self.num):
            x.append(self.points[i][0])
            y.append(self.points[i][1])
        x.append(self.points[0][0])
        y.append(self.points[0][1])
        return x, y
    
    def Tensorize(self):
        tensor=torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0]=self.points[i][0]
            tensor[i][1]=self.points[i][1]
        return tensor
    
    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X)/len(X)
        midY = sum(Y)/len(Y)
        ax.plot(X, Y, linewidth=1)
        ax.text(midX, midY, self.name)


class Dielectric:
    def __init__(self, name, num, points, er):
        self.name = name
        self.num = int(num)
        self.points = points
        self.er = er

    def print_info(self):
        print("dielectric's name: ", self.name)
        print("number of dielectric n: ", self.num)
        print("points: ", self.points)
        print("er: ", self.er)

    def Array(self):
        array=np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0]=self.points[i][0]
            array[i][1]=self.points[i][1]
        return array

    def List_for_Vision(self):
        x = []
        y = []
        for i in range(self.num):
            x.append(self.points[i][0])
            y.append(self.points[i][1])
        x.append(self.points[0][0])
        y.append(self.points[0][1])
        return x, y
    
    def Tensorize(self):
        tensor=torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0]=self.points[i][0]
            tensor[i][1]=self.points[i][1]
        return tensor
    
    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)
        ax.text(midX, midY, self.name)

class Boundary_Polygon:
    def __init__(self, num, points):
        self.num = int(num)
        self.points = points

    def print_info(self):
        print("number of dielectric n: ", self.num)
        print("points: ", self.points)

    def Array(self):
        array=np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0]=self.points[i][0]
            array[i][1]=self.points[i][1]
        return array
    
    def List_for_Vision(self):
        x = []
        y = []
        for i in range(self.num):
            x.append(self.points[i][0])
            y.append(self.points[i][1])
        x.append(self.points[0][0])
        y.append(self.points[0][1])
        return x, y
    
    def Tensorize(self):
        tensor=torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0]=self.points[i][0]
            tensor[i][1]=self.points[i][1]
        return tensor
    
    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)


def read_txt_file(file_path):
    ax_C = plt.figure().add_subplot(111)
    ax_die = plt.figure().add_subplot(111)
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

                k = k+1
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
                    num_of_boundary_polygon_points, boundpoly_points)
                # 打印

                master.print_info()
                master.Visualize(ax_C)
                for env_i in env_list:
                    env_i.print_info()
                    env_i.Visualize(ax_C)
                for dielectric_i in dielectric_list:
                    dielectric_i.print_info()
                    dielectric_i.Visualize(ax_die)
                boundpoly.print_info()
                boundpoly.Visualize(ax_die)

                # 返回结果，可以根据需要修改返回值
                return master, env_list, dielectric_list, boundpoly
            else:
                print("文件行数不足，无法获取必要信息。")
                return None
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


# 调用函数，传入文件路径
file_path = './data/type2_data/BEM_INPUT_1_43817.txt'
# file_path = 'C:\\project\\python\\huadajiutian\\BEM_INPUT_2_43652.txt'
master, env_list, dielectric_list, boundpoly = read_txt_file(file_path)

# 可以根据需要进一步处理 result
tensor=boundpoly.Tensorize()
print(tensor)

# 张量矩阵


# 可视化
plt.show()
