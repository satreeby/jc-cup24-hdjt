import torch
import numpy as np
import matplotlib.pyplot as plt


FACTOR = 100


class Master:
    def __init__(self, name, num, points):
        #直接获得
        self.name = name
        self.num = int(num)
        self.points = points    #从左下角开始
        # 防止不是矩形
        self.shape = 'rectangle'
        self.width = points[1][0] - points[0][0]
        self.height = points[2][1] - points[1][1]
        if num != 4:  # 如果不是矩形，直接用横、竖方向最大范围代替w，h
            xlist = []
            ylist = []
            for i in range(len(points)):
                xlist.append(points[i][0])
                ylist.append(points[i][1])
            self.width = max(xlist) - min(xlist)
            self.heigth = max(ylist) - min(ylist)
            self.shape = 'polygon'
        #需要计算
        self.seq_x = 0
        self.seq_y_list = []

    def print_info(self):
        print("master's name: ", self.name)
        print("shape: ", self.shape)
        print("sequence in conductor layer: ", self.seq_x)
        print("width: ", self.width)
        print("height:", self.height)
        print("master is in layer: ", self.seq_y_list)
        print("number of master points: ", self.num)
        print("points' coordinates: ", self.points)
        print(" ")


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
        tensor = torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0] = self.points[i][0]
            tensor[i][1] = self.points[i][1]
        return tensor

    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)
        ax.text(midX, midY, self.name)


class Bottom:
    def __init__(self, name, num, points):
        #直接获得
        self.name = name
        self.num = int(num)
        self.points = points    #从左下角开始
        self.shape = 'rectangle'
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]

    def print_info(self):
        print("bottom's name: ", self.name)
        print("shape: ", self.shape)
        print("width: ", self.width)
        print("height:", self.height)
        print("number of bottom points: ", self.num)
        print("points' coordinates: ", self.points)
        print(" ")


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
        tensor = torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0] = self.points[i][0]
            tensor[i][1] = self.points[i][1]
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
        # 防止不是矩形
        self.shape = 'rectangle'
        self.width = points[1][0] - points[0][0]
        self.height = points[2][1] - points[1][1]
        if num != 4:  # 如果不是矩形，直接用横、竖方向最大范围代替w，h
            xlist = []
            ylist = []
            for i in range(len(points)):
                xlist.append(points[i][0])
                ylist.append(points[i][1])
            self.width = max(xlist) - min(xlist)
            self.heigth = max(ylist) - min(ylist)
            self.shape = 'polygon'
        # 需要计算
        self.seq_x = 0
        self.seq_y_list = []

    def print_info(self):
        print("env's name: ", self.name)
        print("shape: ", self.shape)
        print("sequence in conductor layer: ", self.seq_x)
        print("width: ", self.width)
        print("height:", self.height)
        print("env is in layer: ", self.seq_y_list)
        print("number of env points: ", self.num)
        print("points' coordinates: ", self.points)
        print(" ")


    def Array(self):
        array = np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0] = self.points[i][0]
            array[i][1] = self.points[i][1]
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
        tensor = torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0] = self.points[i][0]
            tensor[i][1] = self.points[i][1]
        return tensor

    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)
        ax.text(midX, midY, self.name)


class Dielectric:
    def __init__(self, name, num, points, er):
        self.name = name
        self.num = int(num)
        self.points = points
        self.er = er
        #第几个
        self.seq_y = 0
        # 长和宽(有些dielectric不是矩形)
        self.width = 0
        self.height = 0
        if num==4:
            self.width = points[1][0]-points[0][0]
            self.height = points[2][1]-points[1][1]
        else:   # 如果不是矩形，直接用横、竖方向最大范围代替w，h
            xlist = []
            ylist = []
            for i in range(len(points)):
                xlist.append(points[i][0])
                ylist.append(points[i][1])
            self.width = max(xlist)-min(xlist)
            self.heigth = max(ylist)-min(ylist)


    def print_info(self):
        print("dielectric's name: ", self.name)
        print("number of dielectric n: ", self.num)
        print("er: ", self.er)
        print("width: ", self.width)
        print("height: ", self.height)
        print("sequence along y axis: ", self.seq_y)
        print("points: ", self.points)
        print(" ")


    def Array(self):
        array = np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0] = self.points[i][0]
            array[i][1] = self.points[i][1]
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
        tensor = torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0] = self.points[i][0]
            tensor[i][1] = self.points[i][1]
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
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]

    def print_info(self):
        print("number of boundary n: ", self.num)
        print("points: ", self.points)
        print("width: ", self.width)
        print("height ", self.height)
        print(" ")

    def Array(self):
        array = np.zeros((self.num, 2))
        for i in range(self.num):
            array[i][0] = self.points[i][0]
            array[i][1] = self.points[i][1]
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
        tensor = torch.zeros((self.num, 2))
        for i in range(self.num):
            tensor[i][0] = self.points[i][0]
            tensor[i][1] = self.points[i][1]
        return tensor

    def Visualize(self, ax):
        X, Y = self.List_for_Vision()
        midX = sum(X) / len(X)
        midY = sum(Y) / len(Y)
        ax.plot(X, Y, linewidth=1)


class Info_One_input:
    def __init__(self):
        '''定义需要返回的信息'''
        self.master_shape = ''      # 形状
        self.master_seq = 0         # master在导体层的位置
        self.master_width = 0       # master宽度
        self.master_thickness = 0   # master厚度
        self.env_num = 0            # 环境导体个数
        self.env_shape = {}    # 环境导体形状字典
        self.env_seq = {}           # 环境导体位置字典
        self.env_width = {}         # 导体宽度字典
        self.env_thickness = {}     # 导体厚度字典
        self.cond_sep = []          # 导体间距列表
        self.bot_l_space = 0        # layer左侧多出来的距离
        self.bot_r_space = 0        # layer右侧多出来的距离
        self.bot_divider = 0        # layer左右分隔处横坐标
        self.layer_thickness = 0    # layer厚度
        self.cond_to_layer = 0      # 导体底部到layer顶部的距离
        self.boundary_width = 0     # 边界宽
        self.boundary_height = 0    # 边界高
        self.d_width = {}           # 介质宽度字典
        self.d_thickness = {}       # 介质厚度字典
        self.d_Er = {}              # 介质相对介电常数字典
        self.d_layer = {}           # 介质层数字典


class Info_All:
    def __init__(self):
        self.master_shapeL = []  # 形状
        self.master_seqL = []  # master在导体层的位置
        self.master_widthL = []  # master宽度
        self.master_thicknessL = []  # master厚度
        self.env_numL = []  # 环境导体个数
        self.env_shapeL = []  # 环境导体形状字典
        self.env_seqL = []  # 环境导体位置字典
        self.env_widthL = []  # 导体宽度字典
        self.env_thicknessL = []  # 导体厚度字典
        self.cond_sepL = []  # 导体间距字典
        self.bot_l_spaceL = []  # layer左侧多出来的距离
        self.bot_r_spaceL = []  # layer右侧多出来的距离
        self.bot_dividerL = []  # layer左右分隔处横坐标
        self.layer_thicknessL = []  # layer厚度
        self.cond_to_layerL = []  # 导体底部到layer顶部的距离
        self.boundary_widthL = []  # 边界宽
        self.boundary_heightL = []  # 边界高
        self.d_widthL = []  # 介质宽度字典
        self.d_thicknessL = []  # 介质厚度字典
        self.d_ErL = []  # 介质相对介电常数字典
        self.d_layerL = []  # 介质层数字典

    def collect(self, info):
        self.master_shapeL.append(info.master_shape)
        self.master_seqL.append(info.master_seq)
        self.master_widthL.append(info.master_width)
        self.master_thicknessL.append(info.master_thickness)
        self.env_numL.append(info.env_num)
        self.env_shapeL.append(info.env_shape)
        self.env_seqL.append(info.env_seq)
        self.env_widthL.append(info.env_width)
        self.env_thicknessL.append(info.env_thickness)
        self.cond_sepL.append(info.cond_sep)
        self.bot_l_spaceL.append(info.bot_l_space)
        self.bot_r_spaceL.append(info.bot_r_space)
        self.bot_dividerL.append(info.bot_divider)
        self.layer_thicknessL.append(info.layer_thickness)
        self.cond_to_layerL.append(info.cond_to_layer)
        self.boundary_widthL.append(info.boundary_width)
        self.boundary_heightL.append(info.boundary_height)
        self.d_widthL.append(info.d_width)
        self.d_thicknessL.append(info.d_thickness)
        self.d_ErL.append(info.d_Er)
        self.d_layerL.append(info.d_layer)


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
                    k = k + 1
                master = Master(master_name, num_of_mp, points=master_points)

                # env_conductor
                k = k + 1
                num_of_env = int(lines[k].strip())
                k = k + 1
                env_list = []
                bot_list = []
                for _ in range(num_of_env):
                    num_of_env_n = int(lines[k + 1].strip())
                    env_points = []
                    for j1 in range(num_of_env_n):
                        env_points.append([float(num)
                                           for num in lines[k + 2 + j1].split()])
                    name = lines[k].strip()
                    if name == 'botleft' or name == 'botright':
                        bot_instance = Bottom(name=name, num=num_of_env_n, points=env_points)
                        bot_list.append(bot_instance)
                    else:
                        env_instance = Env(name=lines[k].strip(), num=num_of_env_n, points=env_points)
                        env_list.append(env_instance)
                    k = k + 1 + num_of_env_n + 2

                #deilectric
                k = k + 1
                num_of_dielectric = int(lines[k].strip())
                k = k + 1
                dielectric_list = []
                for _ in range(num_of_dielectric):
                    num_of_dielectric_n = int(lines[k + 1].strip())
                    dielectric_points = []
                    for j2 in range(num_of_dielectric_n):
                        dielectric_points.append(
                            [float(num) for num in lines[k + 2 + j2].split()])
                    dielectric_instance = Dielectric(name=lines[k].strip(), num=num_of_dielectric_n,
                        points=dielectric_points, er=int(lines[k + num_of_dielectric_n + 2].strip()))
                    dielectric_list.append(dielectric_instance)
                    k = k + num_of_dielectric_n + 4

                # boundary
                k = k + 1
                num_of_boundary_polygon_points = int(lines[k].strip())
                k = k + 1
                boundpoly_points = []
                for _ in range(num_of_boundary_polygon_points):
                    boundpoly_points.append([float(num)
                                             for num in lines[k].split()])
                    k = k + 1
                boundpoly = Boundary_Polygon(
                    num_of_boundary_polygon_points, boundpoly_points)

                return master, env_list, bot_list, dielectric_list, boundpoly
            else:
                print("文件行数不足，无法获取必要信息。")
                return None
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


def process_data(master, env_list, bot_list, dielectric_list, boundpoly):
    # 需要返回的信息
    info = Info_One_input()

    # 定义画布
    # ax_C = plt.figure().add_subplot(111)
    # ax_die = plt.figure().add_subplot(111)

    # 纵向排布顺序
    die_layer_seq = {}
    for die_i in dielectric_list:
        die_layer_seq[die_i.name] = die_i.points[0][1]  # 左下角纵坐标
    die_sorted = [(key, value) for (key, value) in die_layer_seq.items()]
    die_sorted.sort(key=lambda x: x[1])
    for i in range(len(die_sorted)):
        for d in dielectric_list:
            if d.name == die_sorted[i][0]:
                d.seq_y = i + 1

    # 检查导体占了哪些层
    for d in dielectric_list:
        if abs(d.points[0][1] - master.points[0][1]) < 0.00001 or \
                abs(d.points[3][1] - master.points[3][1]) < 0.00001:  # 同一层(只考虑导体最多占了两行！)
            for e in env_list:
                if not e.seq_y_list:  # 如果空，直接加入
                    e.seq_y_list.append(d.seq_y)
                else:
                    if d.seq_y not in e.seq_y_list:
                        e.seq_y_list.append(d.seq_y)
            if not master.seq_y_list:
                master.seq_y_list.append(d.seq_y)
            else:
                if d.seq_y not in master.seq_y_list:
                    master.seq_y_list.append(d.seq_y)

    # 导体层排布顺序
    con_layer_seq = {}
    con_layer_seq[master.name] = master.points[0][0]  # 左下角横坐标
    for env_i in env_list:
        con_layer_seq[env_i.name] = env_i.points[0][0]
    sorted_item = [(key, value) for (key, value) in con_layer_seq.items()]
    sorted_item.sort(key=lambda x: x[1])  # ???
    for i in range(len(sorted_item)):
        if sorted_item[i][0] == master.name:
            master.seq_x = i + 1
        else:
            for e in env_list:
                if sorted_item[i][0] == e.name:
                    e.seq_x = i + 1

    # 计算separation
    sep_distance = []
    sep_x = {}
    sep_x[str(master.seq_x) + 'l'] = master.points[0][0]
    sep_x[str(master.seq_x) + 'r'] = master.points[1][0]
    for e in env_list:
        sep_x[str(e.seq_x) + 'l'] = e.points[0][0]
        sep_x[str(e.seq_x) + 'r'] = e.points[1][0]
    sep_distance.append(sep_x[str(1) + 'l'] - boundpoly.points[0][0])  # 到左边界的距离
    for i in range(2, 2 + len(env_list)):
        sep_distance.append(sep_x[str(i) + 'l'] - sep_x[str(i - 1) + 'r'])
    sep_distance.append(boundpoly.points[1][0] - sep_x[str(1 + len(env_list)) + 'r'])
    print("最左边导体到左边界的距离：", sep_distance[0])
    for i in range(1, len(sep_distance) - 1):
        print("第" + str(i) + "个导体和第" + str(i + 1) + "个导体间的距离：", sep_distance[i])
    print("最右边导体到右边界的距离：", sep_distance[-1])
    print(" ")

    # 计算导体到bottom的高度(下-上)
    print("导体到bottom的高度：", master.points[0][1] - bot_list[0].points[3][1])

    # layer左右多出来的宽度
    cond_lx = 0  # 导体层最左边的点的横坐标
    cond_rx = 0  # 导体层最右边的点的横坐标
    if master.seq_x == 1:
        cond_lx = master.points[0][0]
    else:
        if master.seq_x == 1 + len(env_list):
            cond_rx = master.points[1][0]
    for e in env_list:
        if e.seq_x == 1:
            cond_lx = e.points[0][0]
        else:
            if e.seq_x == 1 + len(env_list):
                cond_rx = e.points[1][0]
    print("botleft左侧多出来的宽度：", cond_lx - bot_list[0].points[0][0])
    print("botright右侧多出来的宽度：", bot_list[1].points[1][0] - cond_rx)

    # layer左右分界的位置
    print("layer左右分界的x坐标", bot_list[0].points[1][0])
    print(" ")

    # visualize
    master.print_info()
    # master.Visualize(ax_C)
    print("环境导体个数：", len(env_list))
    for env_i in env_list:
        env_i.print_info()
        # env_i.Visualize(ax_C)
    for bot_i in bot_list:
        bot_i.print_info()
        # bot_i.Visualize(ax_C)
    for dielectric_i in dielectric_list:
        dielectric_i.print_info()
        # dielectric_i.Visualize(ax_die)
    boundpoly.print_info()
    # boundpoly.Visualize(ax_die)
    # 可视化
    # plt.show()

    '''对结果保留到小数点后三位'''
    # 返回结果
    info.master_shape = master.shape
    info.master_seq = master.seq_x
    info.master_width = round(master.width, 3)
    info.master_thickness = round(master.height, 3)
    info.env_num = len(env_list)
    for e in env_list:
        info.env_seq[e.name] = e.seq_x
        info.env_shape[e.name] = e.shape
        info.env_width[e.name] = round(e.width, 3)
        info.env_thickness[e.name] = round(e.height, 3)
    for s in range(len(sep_distance)):
        sep_distance[s] = round(sep_distance[s], 3)
    info.cond_sep = sep_distance
    info.bot_l_space = round(cond_lx - bot_list[0].points[0][0], 3)
    info.bot_r_space = round(bot_list[1].points[1][0] - cond_rx, 3)
    info.bot_divider = bot_list[0].points[1][0]
    info.layer_thickness = round(bot_list[0].height, 3)
    info.cond_to_layer = round(master.points[0][1] - bot_list[0].points[3][1], 3)
    info.boundary_width = round(boundpoly.width, 3)
    info.boundary_height = round(boundpoly.height, 3)
    for d in dielectric_list:
        info.d_Er[d.name] = d.er
        info.d_layer[d.name] = d.seq_y
        info.d_width[d.name] = round(d.width, 3)
        info.d_thickness[d.name] = round(d.height, 3)
    return info


def output_info(INFO_ALL, input_num):
    # 打开文件以便写入
    with open('output1.txt', 'w') as f:
        print('Type 1', file=f)
        print('1. master shape:', file=f)
        for i in range(input_num):
            print('input '+str(i+1)+': '+INFO_ALL.master_shapeL[i], file=f)
        print(" ", file=f)
        print('2. master sequence:', file=f)
        for i in range(input_num):
            print('input '+str(i+1)+': '+str(INFO_ALL.master_seqL[i]), file=f)
        print(" ", file=f)
        print('3. master width: ', file=f)
        for i in range(input_num):
            print('input '+str(i+1)+': '+str(INFO_ALL.master_widthL[i]), file=f)
        print(" ", file=f)
        print('4. master thickness: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.master_thicknessL[i]), file=f)
        print(" ", file=f)
        print('5. env number: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.env_numL[i]), file=f)
        print(" ", file=f)
        print('6. env shape: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ':', file=f)
            for key, value in INFO_ALL.env_shapeL[i].items():
                print('    '+key+': '+value, file=f)
        print(" ", file=f)
        print('7. env sequence: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ':', file=f)
            for key, value in INFO_ALL.env_seqL[i].items():
                print('    '+key + ': ' + str(value), file=f)
        print(" ", file=f)
        print('8. env width: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ':', file=f)
            for key, value in INFO_ALL.env_widthL[i].items():
                print('    '+key + ': ' + str(value), file=f)
        print(" ", file=f)
        print('9. env thickness: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ':', file=f)
            for key, value in INFO_ALL.env_thicknessL[i].items():
                print('    '+key + ': ' + str(value), file=f)
        print(" ", file=f)
        print('10. conductor separation: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ':', file=f)
            print("    最左边导体到左边界的距离：", INFO_ALL.cond_sepL[i][0], file=f)
            for j in range(1, len(INFO_ALL.cond_sepL[i]) - 1):
                print("    第" + str(j) + "个导体和第" + str(j + 1) + "个导体间的距离：", INFO_ALL.cond_sepL[i][j], file=f)
            print("    最右边导体到右边界的距离：", INFO_ALL.cond_sepL[i][-1], file=f)
        print(" ", file=f)
        print('11. bottom L space: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.bot_l_spaceL[i]), file=f)
        print(" ", file=f)
        print('12. bottom R space: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.bot_r_spaceL[i]), file=f)
        print(" ", file=f)
        print('13. bottom divider x: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.bot_dividerL[i]), file=f)
        print(" ", file=f)
        print('14. layer thickness: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.layer_thicknessL[i]), file=f)
        print(" ", file=f)
        print('15. distance from conductor to layer: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.cond_to_layerL[i]), file=f)
        print(" ", file=f)
        print('16. boundary width: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.boundary_widthL[i]), file=f)
        print(" ", file=f)
        print('17. boundary height: ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(INFO_ALL.boundary_heightL[i]), file=f)
        print(" ", file=f)
        print('18. dielectric num(including air layer): ', file=f)
        for i in range(input_num):
            print('input ' + str(i+1) + ': ' + str(len(INFO_ALL.d_ErL)), file=f)
        print(" ", file=f)
        print('19. dielectric width: ', file=f)
        for key in INFO_ALL.d_widthL[0].keys():
            print(key + ':', file=f)
            for i in range(input_num):
                print('    input ' + str(i+1) + ': '+str(INFO_ALL.d_widthL[i][key]),file=f)
        print(" ", file=f)
        print('20. dielectric thickness: ', file=f)
        for key in INFO_ALL.d_thicknessL[0].keys():
            print(key + ':', file=f)
            for i in range(input_num):
                print('    input ' + str(i + 1) + ': ' + str(INFO_ALL.d_thicknessL[i][key]), file=f)
        print(" ", file=f)
        print('21. dielectric epsilon r: ', file=f)
        for key in INFO_ALL.d_ErL[0].keys():
            print(key + ':', file=f)
            for i in range(input_num):
                print('    input ' + str(i + 1) + ': ' + str(INFO_ALL.d_ErL[i][key]), file=f)
        print(" ", file=f)
        print('22. dielectric layer num: ', file=f)
        for key in INFO_ALL.d_layerL[0].keys():
            print(key + ':', file=f)
            for i in range(input_num):
                print('    input ' + str(i + 1) + ': ' + str(INFO_ALL.d_layerL[i][key]), file=f)
        print(" ", file=f)


# 所有input文件的信息
INFO_ALL = Info_All()
#读取所有文件
input_num = 64
for i in range(0, input_num):
    file_path = './data/type1_data/BEM_INPUT_'+str(i+1)+'_43652.txt'
    master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
    info = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
    INFO_ALL.collect(info)

output_info(INFO_ALL, input_num)

