import numpy as np
import math
import time


model_type = {
    'c1':0,
    'c1_c2':1,
    'c1_lEnv':2,
    'c1_c2_lEnv_rEnv':3
}



class Master:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points    # points[0] is the coordinate of the bottomleft dot
        self.seq_x = 0
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
    
    def show_points(self):
        print(self.botLeft[0], self.botLeft[1])
        print(self.botRight[0], self.botRight[1])
        print(self.topRight[0], self.topRight[1])
        print(self.topLeft[0], self.topLeft[1])


class Bottom:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]


class Top:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]


class Env:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.seq_x = 0
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2
    

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x: x[0])
        self.topRight = max(top_points, key=lambda x: x[0])
        self.botLeft = min(bot_points, key=lambda x: x[0])
        self.botRight = max(bot_points, key=lambda x: x[0])


class Dielectric:
    def __init__(self, name, num, points, er):
        self.name = name
        self.num = int(num)
        self.points = points
        self.er = er
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x: x[0])
        self.topRight = max(top_points, key=lambda x: x[0])
        self.botLeft = min(bot_points, key=lambda x: x[0])
        self.botRight = max(bot_points, key=lambda x: x[0])


class Boundary_Polygon:
    def __init__(self, num, points):
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]
        


class Info_One_input:
    def __init__(self):
        self.master_top_width = 0       
        self.master_bot_width = 0
        self.master_K = 0           #master斜率
        self.env_top_width = {}         
        self.env_bot_width = {}
        self.env_K = {}             #env斜率
        self.ave_bot_r_y = 0
        self.boundary_lx = 0     
        self.boundary_rx = 0
        self.top_H = 0
        self.metal_H = 0
        self.cond_sep = []
        self.damage_width = []
        self.air_layer_thickness = 0
        self.cond_thickness = 0   #导体层厚度
    

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
                    master_points.append([round(float(num), 6)
                                          for num in lines[k].split()])
                    k = k + 1
                master = Master(master_name, num_of_mp, points=master_points)

                # env_conductor
                k = k + 1
                num_of_env = int(lines[k].strip())
                k = k + 1
                env_list = []
                bot_dict = {}
                top_dict = {}
                for _ in range(num_of_env):
                    num_of_env_n = int(lines[k + 1].strip())
                    env_points = []
                    for j1 in range(num_of_env_n):
                        env_points.append([round(float(num), 6)
                                           for num in lines[k + 2 + j1].split()])
                    name = lines[k].strip()
                    if name == 'botleft' or name == 'botright':
                        bot_instance = Bottom(name=name, num=num_of_env_n, points=env_points)
                        bot_dict[name] = bot_instance
                    elif name == 'topleft' or name == 'topright' or name == 'topcenter':
                        top_instance = Top(name=name, num=num_of_env_n, points=env_points)
                        top_dict[name] = top_instance
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
                            [round(float(num), 6) for num in lines[k + 2 + j2].split()])
                    dielectric_instance = Dielectric(name=lines[k].strip(), num=num_of_dielectric_n,
                        points=dielectric_points, er=round(float(lines[k + num_of_dielectric_n + 2].strip()), 3))
                    dielectric_list.append(dielectric_instance)
                    k = k + num_of_dielectric_n + 4

                # boundary
                k = k + 1
                num_of_boundary_polygon_points = int(lines[k].strip())
                k = k + 1
                boundpoly_points = []
                for _ in range(num_of_boundary_polygon_points):
                    boundpoly_points.append([round(float(num), 6)
                                             for num in lines[k].split()])
                    k = k + 1
                boundpoly = Boundary_Polygon(
                    num_of_boundary_polygon_points, boundpoly_points)

                return master, env_list, bot_dict, top_dict, dielectric_list, boundpoly
            else:
                return None
    except FileNotFoundError:
        print(f"file {file_path} not found")
        return None
    except Exception as e:
        print(f"error: {str(e)}")
        return None


def process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, T, id):
    # info need to be returned
    info = Info_One_input()

    info.master_top_width = round(master.topRight[0]-master.topLeft[0], 6)
    info.master_bot_width = round(master.botRight[0]-master.botLeft[0], 6)
    info.master_K = round((info.master_top_width-info.master_bot_width)/2/master.height, 8)
    info.boundary_lx = boundpoly.points[0][0]
    info.boundary_rx = boundpoly.points[1][0]
    for d in dielectric_list:
        if d.name == 'air_Layer':
            info.air_layer_thickness = round(d.height, 6)
            break
    dnames = []
    for d in dielectric_list:
        if 'damage' in d.name:
            if d.name not in dnames:
                dnames.append(d.name)
                W_top = round(d.topRight[0]-d.topLeft[0], 6)
                W_bot = round(d.botRight[0]-d.botLeft[0], 6)
                if abs(W_top - (info.boundary_rx-info.boundary_lx)) > 0.001:    # 有一个damage撑满了，这个不算
                    info.damage_width.append(W_top)
                if abs(W_bot - (info.boundary_rx-info.boundary_lx)) > 0.001:
                    info.damage_width.append(W_bot)
    #damage: PLAET2、3L有2中共，STACK3L有4种
    if T == 'PLATE2L' or T == 'PLATE3L':
        if 4-len(info.damage_width) > 0:
            for i in range(4-len(info.damage_width)):
                info.damage_width.append(0)
    if T == 'STACK3L':
        if 8-len(info.damage_width) > 0:
            for i in range(8-len(info.damage_width)):
                info.damage_width.append(0)
    # 环境导体的处理需要分type
    if T == 'PLATE2L' or T == 'PLATE3L':
        for e in env_list:
            if e.name == 'lEnv':
                info.env_top_width['lEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['lEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['lEnv'] = round((info.env_top_width['lEnv']-info.env_bot_width['lEnv'])/2/e.height, 8)
            if e.name == 'c2':
                info.env_top_width['c2'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['c2'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['c2'] = round((info.env_top_width['c2']-info.env_bot_width['c2'])/2/e.height, 8)
            if e.name == 'rEnv':
                info.env_top_width['rEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['rEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['rEnv'] = round((info.env_top_width['rEnv']-info.env_bot_width['rEnv'])/2/e.height, 8)
    elif T == 'STACK3L':
        for e in env_list:
            if e.name == 'lbEnv':
                info.env_top_width['lbEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['lbEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['lbEnv'] = round((info.env_top_width['lbEnv']-info.env_bot_width['lbEnv'])/2/e.height, 8)
            if e.name == 'c3':
                info.env_top_width['c3'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['c3'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['c3'] = round((info.env_top_width['c3']-info.env_bot_width['c3'])/2/e.height, 8)
            if e.name == 'rbEnv':
                info.env_top_width['rbEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['rbEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['rbEnv'] = round((info.env_top_width['rbEnv']-info.env_bot_width['rbEnv'])/2/e.height, 8)
            if e.name == 'ltEnv':
                info.env_top_width['ltEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['ltEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['ltEnv'] = round((info.env_top_width['ltEnv']-info.env_bot_width['ltEnv'])/2/e.height, 8)
            if e.name == 'c4':
                info.env_top_width['c4'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['c4'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['c4'] = round((info.env_top_width['c4']-info.env_bot_width['c4'])/2/e.height, 8)
            if e.name == 'rtEnv':
                info.env_top_width['rtEnv'] = round(e.topRight[0]-e.topLeft[0], 6)
                info.env_bot_width['rtEnv'] = round(e.botRight[0]-e.botLeft[0], 6)
                info.env_K['rtEnv'] = round((info.env_top_width['rtEnv']-info.env_bot_width['rtEnv'])/2/e.height, 8)
    # 导体层间距
    if T == 'PLATE2L' or T == 'PLATE3L':
        env_names = []
        for e in env_list:
            env_names.append(e.name)
        if 'lEnv' in env_names:
            for e in env_list:
                if e.name == 'lEnv':
                    info.cond_sep.append(round(master.topLeft[0]-e.topRight[0], 6))
                    info.cond_sep.append(round(master.botLeft[0]-e.botRight[0], 6))
        if 'c2' in env_names:      # 只要c2不在，rEnv一定不在
            for e in env_list:
                if e.name == 'c2':
                    c2_topRx = e.topRight[0]
                    c2_botRx = e.botRight[0]
                    info.cond_sep.append(round(e.topLeft[0]-master.topRight[0], 6))
                    info.cond_sep.append(round(e.botLeft[0]-master.botRight[0], 6))
            if 'rEnv' in env_names:
                for e in env_list:
                    if e.name == 'rEnv':
                        info.cond_sep.append(round(e.topLeft[0]-c2_topRx, 6))
                        info.cond_sep.append(round(e.botLeft[0]-c2_botRx, 6))
    elif T == 'STACK3L':
        if id%2 == 1:
            # 下层
            env_names = []
            for e in env_list:
                env_names.append(e.name)
            if 'lbEnv' in env_names:
                for e in env_list:
                    if e.name == 'lbEnv':
                        info.cond_sep.append(round(master.topLeft[0]-e.topRight[0], 6))
                        info.cond_sep.append(round(master.botLeft[0]-e.botRight[0], 6))
            if 'c3' in env_names:      # 只要c2不在，rEnv一定不在
                for e in env_list:
                    if e.name == 'c3':
                        c3_topRx = e.topRight[0]
                        c3_botRx = e.botRight[0]
                        info.cond_sep.append(round(e.topLeft[0]-master.topRight[0], 6))
                        info.cond_sep.append(round(e.botLeft[0]-master.botRight[0], 6))
                if 'rbEnv' in env_names:
                    for e in env_list:
                        if e.name == 'rbEnv':
                            info.cond_sep.append(round(e.topLeft[0]-c3_topRx, 6))
                            info.cond_sep.append(round(e.botLeft[0]-c3_botRx, 6))
        else:
        # 上层
            env_names = []
            for e in env_list:
                env_names.append(e.name)
            if 'ltEnv' in env_names:
                for e in env_list:
                    if e.name == 'ltEnv':
                        info.cond_sep.append(round(master.topLeft[0]-e.topRight[0], 6))
                        info.cond_sep.append(round(master.botLeft[0]-e.botRight[0], 6))
            if 'c4' in env_names:      # 只要c2不在，rEnv一定不在
                for e in env_list:
                    if e.name == 'c4':
                        c4_topRx = e.topRight[0]
                        c4_botRx = e.botRight[0]
                        info.cond_sep.append(round(e.topLeft[0]-master.topRight[0], 6))
                        info.cond_sep.append(round(e.botLeft[0]-master.botRight[0], 6))
                if 'rtEnv' in env_names:
                    for e in env_list:
                        if e.name == 'rtEnv':
                            info.cond_sep.append(round(e.topLeft[0]-c4_topRx, 6))
                            info.cond_sep.append(round(e.botLeft[0]-c4_botRx, 6))
    
    # metal_H需要分type
    info.metal_H = round(master.botLeft[1]-bot_dict['botleft'].points[3][1], 6)

    if T == 'PLATE3L':
        if 'topright' in top_dict:
            info.top_H = round(top_dict['topright'].points[0][1]-master.topLeft[1], 6)
    
    #导体层厚度
    info.cond_thickness = round(master.height, 6)

    return info


def output(info, id, T, model_type):
    # 'c1':0,
    # 'c1_c2':1,
    # 'c1_lEnv':2,
    # 'c1_c2_lEnv_rEnv':3
    re_info = []

    if T == 'PLATE2L':
        if model_type == 0: #c1
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.metal_H)
            re_info.append(info.cond_thickness)
            for dw in info.damage_width:
                re_info.append(dw)
        elif model_type == 2: 
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            re_info.append(math.log(info.master_top_width+1))#将大值缩小，小值不变，不改变大小关系
            re_info.append(math.log(info.master_bot_width+1))
            re_info.append(info.master_K)
            re_info.append(math.log(1+info.cond_thickness/info.metal_H))#这个单独对不需要求c12的情况进行处理
            for value in info.env_top_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_bot_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_K.values():
                re_info.append(value)
            for sep in info.cond_sep:
                re_info.append(math.log(sep+1))
            for dw in info.damage_width:
                re_info.append(dw)
        elif model_type == 3:
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            re_info.append(math.log(info.master_top_width+1))
            re_info.append(math.log(info.master_bot_width+1))
            re_info.append(info.master_K)
            re_info.append(info.metal_H)
            re_info.append(info.cond_thickness)
            #单独处理梯形斜边正对电容
            theta = 2*math.atan(info.master_K)
            l = info.cond_thickness/math.cos(theta/2)
            l0 = info.cond_sep[1]/math.sin(theta/2)
            re_info.append((math.log(1+l/l0))/theta)
            #
            for value in info.env_top_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_bot_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_K.values():
                re_info.append(value)
            for i in range(len(info.cond_sep)):
                if i == 2 or i == 3:
                    re_info.append(math.sqrt(info.cond_sep[i]))
                else:
                    re_info.append(math.log(info.cond_sep[i]+1))
            for dw in info.damage_width:
                re_info.append(dw)
        elif model_type == 1:
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            re_info.append(math.log(info.master_top_width+1))
            re_info.append(math.log(info.master_bot_width+1))
            re_info.append(info.master_K)
            re_info.append(info.metal_H)
            re_info.append(info.cond_thickness)
            # re_info.append(math.log(1+info.cond_thickness/info.metal_H))
            # re_info.append(math.log(1+(info.env_top_width+info.env_bot_width)/((info.cond_sep[0]+info.cond_sep[1])/2)))
            #单独处理梯形斜边正对电容
            theta = 2*math.atan(info.master_K)
            l = info.cond_thickness/math.cos(theta/2)
            l0 = info.cond_sep[1]/math.sin(theta/2)
            re_info.append((math.log(1+l/l0))/theta)
            #
            for value in info.env_top_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_bot_width.values():
                re_info.append(math.log(value+1))
            for value in info.env_K.values():
                re_info.append(value)
            for i in range(len(info.cond_sep)):
                if i == 2 or i == 3:
                    re_info.append(math.sqrt(info.cond_sep[i]))
                else:
                    re_info.append(math.log(info.cond_sep[i]+1))
            for dw in info.damage_width:
                re_info.append(dw)
    elif T == 'PLATE3L':
        if model_type == 0:
            if id % 2 == 1:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.metal_H)
                re_info.append(info.top_H)
                re_info.append(info.cond_thickness)
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.top_H)
        elif model_type == 1 or model_type == 3: 
            if id % 2 == 0:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.metal_H)
                re_info.append(info.top_H)
                re_info.append(info.cond_thickness)
                for value in info.env_top_width.values():
                    re_info.append(value)
                for value in info.env_bot_width.values():
                    re_info.append(value)
                for value in info.env_K.values():
                    re_info.append(value)
                for sep in info.cond_sep:
                    re_info.append(math.log(sep+1))
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.top_H)
        elif model_type == 2: 
            if id % 2 == 0:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.cond_thichness/info.metal_H)
                re_info.append(info.cond_thichness/info.top_H)
                for value in info.env_top_width.values():
                    re_info.append(value)
                for value in info.env_bot_width.values():
                    re_info.append(value)
                for value in info.env_K.values():
                    re_info.append(value)
                for sep in info.cond_sep:
                    re_info.append(math.log(sep+1))
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.top_H)
    elif T == 'STACK3L':
        if model_type == 0:
            if id % 2 == 1:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.metal_H)
                re_info.append(info.cond_thickness)
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.cond_thickness)
        elif model_type == 1 or model_type == 3: 
            if id % 2 == 0:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.metal_H)
                re_info.append(info.top_H)
                re_info.append(info.cond_thickness)
                for value in info.env_top_width.values():
                    re_info.append(value)
                for value in info.env_bot_width.values():
                    re_info.append(value)
                for value in info.env_K.values():
                    re_info.append(value)
                for sep in info.cond_sep:
                    re_info.append(math.log(sep+1))
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.top_H)
        elif model_type == 2: 
            if id % 2 == 0:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.cond_thichness/info.metal_H)
                re_info.append(info.cond_thichness/info.top_H)
                for value in info.env_top_width.values():
                    re_info.append(value)
                for value in info.env_bot_width.values():
                    re_info.append(value)
                for value in info.env_K.values():
                    re_info.append(value)
                for sep in info.cond_sep:
                    re_info.append(math.log(sep+1))
                for dw in info.damage_width:
                    re_info.append(dw)
            else:
                re_info.append(info.master_top_width)
                re_info.append(info.top_H)

    return re_info
 

def parser_features(pattern,     
           metal, 
           pattern_path, # pattern 所在文件夹路径
           generation,   #是否是生成的数据
           input_num,    #生成的数据需要指定输入文件数量
           single_file,  #是否是单个文件
           single_file_path1,      #单个文件编号
           single_file_path2
           ):
    
    # 'c1':0,
    # 'c1_c2':1,
    # 'c1_lEnv':2,
    # 'c1_c2_lEnv_rEnv':3
    feature_matrix = [[], [], [], []]
    model_index = [[], [], [], []]
    assert pattern == 'PLATE2L' or pattern == 'PLATE3L' or pattern == 'STACK3L', 'pattern error!'

    if single_file:
        if pattern == 'PLATE2L':
            #read
            master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(single_file_path1)
            # 判断model type
            if len(env_list) == 0:
                T = model_type['c1']
            elif len(env_list) == 3:
                T = model_type['c1_c2_lEnv_rEnv']
            elif len(env_list) == 1:
                if env_list[0].name == 'c2':
                    T = model_type['c1_c2']
                if env_list[0].name == 'lEnv':
                    T = model_type['c1_lEnv']
            #process
            info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, 'PLATE2L', 1)
            feature_list = output(info, 1, 'PLATE2L', T)
        if pattern == 'PLATE3L':
            master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1 = read_txt_file(single_file_path1)
            master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2 = read_txt_file(single_file_path2)
            # 判断model type
            if len(env_list1) == 0:
                T = model_type['c1']
            elif len(env_list1) == 3:
                T = model_type['c1_c2_lEnv_rEnv']
            elif len(env_list1) == 1:
                if env_list1[0].name == 'c2':
                    T = model_type['c1_c2']
                if env_list1[0].name == 'lEnv':
                    T = model_type['c1_lEnv']
            info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, 'PLATE3L', 1)
            info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, 'PLATE3L', 2)
            re1 = output(info1, 1, 'PLATE3L', T)
            re2 = output(info2, 2, 'PLATE3L', T)
            feature_list = re1+re2
        if pattern == 'STACK3L':
            master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1 = read_txt_file(single_file_path1)
            master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2 = read_txt_file(single_file_path2)
            # 判断model type
            if len(env_list1) == 2:
                T = model_type['c1']
            elif len(env_list1) == 8:
                T = model_type['c1_c2_lEnv_rEnv']
            elif len(env_list1) == 4:
                for env in env_list1:
                    if env.name == 'c3' or env.name == 'c4':
                        T = model_type['c1_c2']
                        break
                    if env.name == 'lbEnv' or env.name == 'ltEnv':
                        T = model_type['c1_lEnv']
                        break
            info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, 'PLATE3L', 1)
            info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, 'PLATE3L', 2)
            re1 = output(info1, 1, 'PLATE3L', T)
            re2 = output(info2, 2, 'PLATE3L', T)
        return feature_list, T
    #整个文件夹一起读
    else:
        if generation:
            if pattern == 'PLATE2L':
                for i in range(0, input_num):
                    file_path = pattern_path+"/file_"+str(i+1)+".txt"
                    #read
                    master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(file_path)
                    # 判断model type
                    if len(env_list) == 0:
                        T = model_type['c1']
                    elif len(env_list) == 3:
                        T = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list) == 1:
                        if env_list[0].name == 'c2':
                            T = model_type['c1_c2']
                        if env_list[0].name == 'lEnv':
                            T = model_type['c1_lEnv']
                    #提取信息
                    info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, i+1)
                    feature_matrix[T].append(output(info, i+1, pattern, T))
                    model_index[T].append(i)
            if pattern == 'PLATE3L':
                assert input_num%2==0, 'input number error!'
                input_num = input_num/2

                for i in range(0, input_num):
                    c1_id = 2*i+1
                    botcenter_id = 2*i+2
                    file1_path = pattern_path+"/file_"+str(c1_id)+".txt"
                    file2_path = pattern_path+"/file_"+str(botcenter_id)+".txt"
                    #read
                    master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1 = read_txt_file(file1_path)
                    master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2 = read_txt_file(file2_path)
                    # 判断model type
                    if len(env_list1) == 0:
                        T = model_type['c1']
                    elif len(env_list1) == 3:
                        T = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list1) == 1:
                        if env_list1[0].name == 'c2':
                            T = model_type['c1_c2']
                        if env_list1[0].name == 'lEnv':
                            T = model_type['c1_lEnv']
                    #提取信息
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern, c1_id)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern, botcenter_id)
                    re1 = output(info1, c1_id, pattern, T)
                    re2 = output(info2, botcenter_id, pattern, T)
                    re = re1+re2
                    feature_matrix[T].append(re)
                    model_index[T].append(i)

            if pattern == 'STACK3L':
                assert input_num%2==0, 'input number error!'
                input_num = input_num/2

                for i in range(0, input_num):
                    c1_id = 2*i+1
                    c2_id = 2*i+2
                    file1_path = pattern_path+"/file_"+str(c1_id)+".txt"
                    file2_path = pattern_path+"/file_"+str(c2_id)+".txt"
                    #read
                    master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1 = read_txt_file(file1_path)
                    master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2 = read_txt_file(file2_path)
                    # 判断model type
                    if len(env_list1) == 2:
                        T = model_type['c1']
                    elif len(env_list1) == 8:
                        T = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list1) == 4:
                        for env in env_list1:
                            if env.name == 'c3' or env.name == 'c4':
                                T = model_type['c1_c2']
                                break
                            if env.name == 'lbEnv' or env.name == 'ltEnv':
                                T = model_type['c1_lEnv']
                        break
                    #提取信息
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern, c1_id)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern, c2_id)
                    re1 = output(info1, c1_id, pattern, T)
                    re2 = output(info2, c2_id, pattern, T)
                    re = re1+re2
                    feature_matrix[T].append(re)
                    model_index[T].append(i)
        else:
            if pattern == 'PLATE2L':
                assert metal == '1' or metal == '2' or metal == '3', 'metal error!'

                input_num = 324
                for i in range(0, input_num):
                    if metal == '1':
                        file_path = pattern_path+'/PLATE2L/SUB-metal1_PLATE2L/input/BEM_INPUT_'+str(i+1)+'_131919.txt'
                    if metal == '2':
                        file_path = pattern_path+'/PLATE2L/SUB-metal2_PLATE2L/input/BEM_INPUT_'+str(i+1)+'_132273.txt'
                    if metal == '3':
                        file_path = pattern_path+'/PLATE2L/SUB-metal3_PLATE2L/input/BEM_INPUT_'+str(i+1)+'_132771.txt'
                    #read
                    master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(file_path)
                    # 判断model type
                    if len(env_list) == 0:
                        T = model_type['c1']
                    elif len(env_list) == 3:
                        T = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list) == 1:
                        if env_list[0].name == 'c2':
                            T = model_type['c1_c2']
                        if env_list[0].name == 'lEnv':
                            T = model_type['c1_lEnv']
                    #提取信息
                    info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, 'PLATE2L', i+1)
                    feature_matrix[T].append(output(info, i+1, 'PLATE2L', T))
                    model_index[T].append(i)
            
            elif pattern == 'PLATE3L':
                assert metal == '12' or metal == '13' or metal == '23', 'metal error!'

                input_num = 324
                for i in range(0, input_num):
                    c1_id = 2*i+1
                    botcenter_id = 2*i+2
                    if metal == '12':
                        file1_path = pattern_path+'/PLATE3L/SUB-metal1-metal2_PLATE3L/input/BEM_INPUT_'+str(c1_id)+'_133293.txt'
                        file2_path = pattern_path+'/PLATE3L/SUB-metal1-metal2_PLATE3L/input/BEM_INPUT_'+str(botcenter_id)+'_133293.txt'
                    if metal == '13':
                        file1_path = pattern_path+'/PLATE3L/SUB-metal1-metal3_PLATE3L/input/BEM_INPUT_'+str(c1_id)+'_138255.txt'
                        file2_path = pattern_path+'/PLATE3L/SUB-metal1-metal3_PLATE3L/input/BEM_INPUT_'+str(botcenter_id)+'_138255.txt'
                    if metal == '23':
                        file1_path = pattern_path+'/PLATE3L/SUB-metal2-metal3_PLATE3L/input/BEM_INPUT_'+str(c1_id)+'_133631.txt'
                        file2_path = pattern_path+'/PLATE3L/SUB-metal2-metal3_PLATE3L/input/BEM_INPUT_'+str(botcenter_id)+'_133631.txt'
                    #read
                    master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1 = read_txt_file(file1_path)
                    master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2 = read_txt_file(file2_path)
                    # 判断model type
                    if len(env_list1) == 0:
                        T = model_type['c1']
                    elif len(env_list1) == 3:
                        T = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list1) == 1:
                        if env_list1[0].name == 'c2':
                            T = model_type['c1_c2']
                        if env_list1[0].name == 'lEnv':
                            T = model_type['c1_lEnv']
                    #提取信息
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, 'PLATE3L', c1_id)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, 'PLATE3L', botcenter_id)
                    re1 = output(info1, c1_id, 'PLATE3L', T)
                    re2 = output(info2, botcenter_id, 'PLATE3L', T)
                    re = re1+re2
                    feature_matrix[T].append(re)
                    model_index[T].append(i)
            
            elif pattern == 'STACK3L':
                assert metal == '12' or metal == '13' or metal == '23', 'metal error!'

                input_num = 144
                for i in range(0, input_num):
                    c1_id = 2*i+1
                    c2_id = 2*i+2
                    if metal == '12':
                        file1_path = pattern_path+'/STACK3L/SUB-metal1-metal2_STACK3L/input/BEM_INPUT_'+str(c1_id)+'_129796.txt'
                        file2_path = pattern_path+'/STACK3L/SUB-metal1-metal2_STACK3L/input/BEM_INPUT_'+str(c2_id)+'_129796.txt'
                    if metal == '13':
                        file1_path = pattern_path+'/STACK3L/SUB-metal1-metal3_STACK3L/input/BEM_INPUT_'+str(c1_id)+'_131209.txt'
                        file2_path = pattern_path+'/STACK3L/SUB-metal1-metal3_STACK3L/input/BEM_INPUT_'+str(c2_id)+'_131209.txt'
                    if metal == '23':
                        file1_path = pattern_path+'/STACK3L/SUB-metal2-metal3_STACK3L/input/BEM_INPUT_'+str(c1_id)+'_126367.txt'
                        file2_path = pattern_path+'/STACK3L/SUB-metal2-metal3_STACK3L/input/BEM_INPUT_'+str(c2_id)+'_126367.txt'
                    master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
                    master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, 'STACK3L', c1_id)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, 'STACK3L', c2_id)
                    re1 = output(info1, c1_id, 'STACK3L', T)
                    re2 = output(info2, c2_id, 'STACK3L', T)
                    re = re1+re2
                    feature_matrix[T].append(re)
                    model_index[T].append(i)

        return feature_matrix, model_index


if __name__ == '__main__':
    start = time.time()

    '''Usage（STACK3L先别用）'''
    #首先定义导体类型如下：
    #PLATE2L  &&  PLATE3L
    # 'c1': 0,
    # 'c1_c2': 1,
    # 'c1_lEnv': 2,
    # 'c1_c2_lEnv_rEnv': 3
    '''input'''
    #如果所有文件一起读，single_file=False，并需要指定参数pattern，metal，pattern_path。pattern='PLATE2L'时metal='1' or '2' or '3', pattern='PLATE3L'时metal='12'or'13'or'23', pattern='STACK3L'时metal = '12'or'13'or'23
                                                            # 另外，如果是华大九天的数据，generation=False，pattern_path是Cases文件夹路径，最后三个参数不用管，如果是自己生成的数据，generation=True，pattern_path是file.txt所在文件夹路径，并指定文件数量input_num
    #如果读入单个文件，single_file=True，并需要指定参数pattern。如果是PLATE2L，需要指定文件路径single_file_path1
                                                            #  如果是PLATE3L，需要指定single_file_path1为以c1为master的文件路径，single_file_path2为以botcenter为master的文件路径，一起输入
    '''output'''
    #如果是读入所有文件，返回feature_matrix和model_index，其中包括了：
    #对于所有只含c1的采样点的【特征矩阵】与对应【文件编号list】，在feature_matrix[0]和model_index[0]中。feature_matrix[0]是一个二维矩阵，model_index是一维list，其中的文件编号从0开始数
    #只含c1_c2    --->   feature[1], model_index[1]
    #只含c1_lEnv  --->   feature[2], model_index[2]
    #含所有导体    --->   feature[3], model_index[3]

    #如果是读入单个文件，返回该文件的特征（一维list）以及所属的类型
    #注：使用feature_matrix[i]和model_index[i]的时候需要检查其是否为空
    
    feature_matrix_metal3, model_index_metal3 = parser_features(pattern='PLATE2L', metal='3', 
                                         pattern_path='./Cases',
                                         generation=False, input_num=324,
                                         single_file=False, 
                                         single_file_path1='',
                                         single_file_path2=''
                                         )
    with open('./test_parser.txt', 'w') as f:
        print(len(feature_matrix_metal3[3]), file=f)
        print("", file=f)
        print(len(model_index_metal3[3]), file=f)

    end = time.time()
    print(end-start)


    # print(feature_matrix)
    





