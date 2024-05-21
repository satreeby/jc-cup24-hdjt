import numpy as np
import math
import time

'''suppose:
1. PLATE2L的环境导体的名称只能是lEnv, rEnv, c2, botleft, botright，不能出现其它名称的环境导体，否则无法处理
   PLATE3L中，以c1为master的情况下环境导体的名称只能是lEnv, rEnv, c2, botleft, botright, topleft, topright
              以botcenter为master的情况下环境导体的名称只能是c1, lEnv, rEnv, c2, botleft, botright, topleft, topright, topcenter
   STACK3L中，以c1为master的情况下环境导体的名称只能是lbEnv, rbEnv, ltEnv, rtEnv, c2, c2left, c3, c4, botleft, botright
              以c2为master的情况下环境导体的名称只能是lbEnv, rbEnv, ltEnv, rtEnv, c1, c1left, c3, c4, botleft, botright
2. 对于PLATE3L，需要连续读入两个相邻的文件，这两个文件中导体分布类似，只是一个是c1为master，一个是botcenter为master
   对于STACK3L同理，只是一个是c1为master，一个是c2为master
'''


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
        self.master_K = 0
        self.env_top_width = {}         
        self.env_bot_width = {}
        self.env_K = {}
        self.metal_top = 0
        self.boundary_lx = 0     
        self.boundary_rx = 0
        self.top_H = 0
        self.metal_H = 0
        self.cond_sep = []
        self.damage_width = []
        self.cond_thickness = 0


def Identify_pattern(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly):
    # 判断pattern
    pattern = 'PLATE2L'
    if len(top_dict) != 0:
        pattern = 'PLATE3L'
    else:
        for e in env_list:
            if e.botLeft[1] > master.topLeft[1] or e.topLeft[1] < master.botLeft[1]:
                pattern = "STACK3L"
                break
    return pattern


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
                
                #identify pattern 
                pattern = Identify_pattern(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly)

                return master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern
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

    info.master_top_width = master.topRight[0]-master.topLeft[0]
    info.master_bot_width = master.botRight[0]-master.botLeft[0]
    info.master_K = (info.master_top_width-info.master_bot_width)/2/master.height
    info.boundary_lx = boundpoly.points[0][0]
    info.boundary_rx = boundpoly.points[1][0]

    dnames = []
    for d in dielectric_list:
        if 'damage' in d.name:
            if d.name not in dnames:
                dnames.append(d.name)
                W_top = d.topRight[0]-d.topLeft[0]
                W_bot = d.botRight[0]-d.botLeft[0]
                info.damage_width.append(W_top)
                info.damage_width.append(W_bot)
    if T == 'PLATE2L':
        if 4-len(info.damage_width) > 0:
            for i in range(4-len(info.damage_width)):
                info.damage_width.append(0.0)
        elif 4-len(info.damage_width) < 0:
            while len(info.damage_width)!=4:
                info.damage_width.pop()
    if T == 'PLATE3L':
        if 6-len(info.damage_width) > 0:
            for i in range(6-len(info.damage_width)):
                info.damage_width.append(0.0)
        elif 6-len(info.damage_width) < 0:
            while len(info.damage_width)!=6:
                info.damage_width.pop()
    if T == 'STACK3L':
        if 8-len(info.damage_width) > 0:
            for i in range(8-len(info.damage_width)):
                info.damage_width.append(0.0)
        elif 8-len(info.damage_width) < 0:
            while len(info.damage_width)!=8:
                info.damage_width.pop()
    # 环境导体的处理需要分type
    if T == 'PLATE2L':
        WlEnv_top = 0
        WlEnv_bot = 0
        Wc2_top = 0
        Wc2_bot = 0
        WrEnv_top = 0
        WrEnv_bot = 0
        Kc2 = 0
        KlEnv = 0
        KrEnv = 0
        for e in env_list:
            if e.name == 'lEnv':
                WlEnv_top = e.topRight[0]-e.topLeft[0]
                WlEnv_bot = e.botRight[0]-e.botLeft[0]
                KlEnv = (WlEnv_top-WlEnv_bot)/2/e.height
            if e.name == 'c2':
                Wc2_top = e.topRight[0]-e.topLeft[0]
                Wc2_bot = e.botRight[0]-e.botLeft[0]
                Kc2 = (Wc2_top-Wc2_bot)/2/e.height
            if e.name == 'rEnv':
                WrEnv_top = e.topRight[0]-e.topLeft[0]
                WrEnv_bot = e.botRight[0]-e.botLeft[0]
                KrEnv = (WrEnv_top-WrEnv_bot)/2/e.height
        info.env_top_width['lEnv'] = WlEnv_top
        info.env_bot_width['lEnv'] = WlEnv_bot
        info.env_K['lEnv'] = KlEnv
        info.env_top_width['c2'] = Wc2_top
        info.env_bot_width['c2'] = Wc2_bot
        info.env_K['c2'] = Kc2
        info.env_top_width['rEnv'] = WrEnv_top
        info.env_bot_width['rEnv'] = WrEnv_bot
        info.env_K['rEnv'] = KrEnv
    elif T == 'PLATE3L':
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
        WlbEnv_top = 0
        WlbEnv_bot = 0
        Wc3_top = 0
        Wc3_bot = 0
        WrbEnv_top = 0
        WrbEnv_bot = 0
        WltEnv_top = 0
        WltEnv_bot = 0
        Wc4_top = 0
        Wc4_bot = 0
        WrtEnv_top = 0
        WrtEnv_bot = 0
        KlbEnv = 0
        Kc3 = 0
        KrbEnv = 0
        KltEnv = 0
        Kc4 = 0
        KrtEnv = 0
        for e in env_list:
            if e.name == 'lbEnv':
                WlbEnv_top = e.topRight[0]-e.topLeft[0]
                WlbEnv_bot = e.botRight[0]-e.botLeft[0]
                KlbEnv = (WlbEnv_top-WlbEnv_bot)/2/e.height
            if e.name == 'c3':
                Wc3_top = e.topRight[0]-e.topLeft[0]
                Wc3_bot = e.botRight[0]-e.botLeft[0]
                Kc3 = (Wc3_top-Wc3_bot)/2/e.height
            if e.name == 'rbEnv':
                WrbEnv_top = e.topRight[0]-e.topLeft[0]
                WrbEnv_bot = e.botRight[0]-e.botLeft[0]
                KrbEnv = (WrbEnv_top-WrbEnv_bot)/2/e.height
            if e.name == 'ltEnv':
                WltEnv_top = e.topRight[0]-e.topLeft[0]
                WltEnv_bot = e.botRight[0]-e.botLeft[0]
                KltEnv = (WltEnv_top-WlbEnv_bot)/2/e.height
            if e.name == 'c4':
                Wc4_top = e.topRight[0]-e.topLeft[0]
                Wc4_bot = e.botRight[0]-e.botLeft[0]
                Kc4 = (Wc4_top-Wc4_bot)/2/e.height
            if e.name == 'rtEnv':
                WrtEnv_top = e.topRight[0]-e.topLeft[0]
                WrtEnv_bot = e.botRight[0]-e.botLeft[0]
                KrtEnv = (WrtEnv_top-WrtEnv_bot)/2/e.height
        info.env_top_width['lbEnv'] = WlbEnv_top
        info.env_bot_width['lbEnv'] = WlbEnv_bot
        info.env_top_width['c3'] = Wc3_top
        info.env_bot_width['c3'] = Wc3_bot
        info.env_top_width['rbEnv'] = WrbEnv_top
        info.env_bot_width['rbEnv'] = WrbEnv_bot
        info.env_top_width['ltEnv'] = WltEnv_top
        info.env_bot_width['ltEnv'] = WltEnv_bot
        info.env_top_width['c4'] = Wc4_top
        info.env_bot_width['c4'] = Wc4_bot
        info.env_top_width['rtEnv'] = WrtEnv_top
        info.env_bot_width['rtEnv'] = WrtEnv_bot
        info.env_K['lbEnv'] = KlbEnv
        info.env_K['c3'] = Kc3
        info.env_K['rbEnv'] = KrbEnv
        info.env_K['ltEnv'] = KltEnv
        info.env_K['c4'] = Kc4
        info.env_K['rtEnv'] = KrtEnv
    # 导体层间距
    if T == 'PLATE2L':
        env_names = []
        for e in env_list:
            env_names.append(e.name)
        if 'lEnv' in env_names:
            for e in env_list:
                if e.name == 'lEnv':
                    info.cond_sep.append(master.botLeft[0]-e.botRight[0])
                    info.cond_sep.append(master.topLeft[0]-e.topRight[0])
        else:
            info.cond_sep.append(0.0)
            info.cond_sep.append(0.0)
        if 'c2' not in env_names:      # 只要c2不在，rEnv一定不在
            info.cond_sep.append(0.0)
            info.cond_sep.append(0.0)
            info.cond_sep.append(0.0)
            info.cond_sep.append(0.0)
        else:
            for e in env_list:
                if e.name == 'c2':
                    c2_rtx = e.topRight[0]
                    c2_rbx = e.botRight[0]
                    info.cond_sep.append(e.botLeft[0]-master.botRight[0])
                    info.cond_sep.append(e.topLeft[0]-master.topRight[0])
            if 'rEnv' not in env_names:
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
            else:
                for e in env_list:
                    if e.name == 'rEnv':
                        info.cond_sep.append(e.botLeft[0]-c2_rbx)
                        info.cond_sep.append(e.topLeft[0]-c2_rtx)
    elif T == 'PLATE3L':
        env_names = []
        for e in env_list:
            env_names.append(e.name)
        if 'lEnv' in env_names:
            for e in env_list:
                if e.name == 'lEnv':
                    info.cond_sep.append(round(master.topLeft[0]-e.topRight[0], 6))
                    info.cond_sep.append(round(master.botLeft[0]-e.botRight[0], 6))
        if 'c2' in env_names: # 只要c2不在，rEnv一定不在
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
                        info.cond_sep.append(master.botLeft[0]-e.botRight[0])
                        info.cond_sep.append(master.topLeft[0]-e.topRight[0])
            else:
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
            if 'c3' not in env_names:      # 只要c2不在，rEnv一定不在
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
            else:
                for e in env_list:
                    if e.name == 'c3':
                        c3_rtx = e.topRight[0]
                        c3_rbx = e.botRight[0]
                        info.cond_sep.append(e.botLeft[0]-master.botRight[0])
                        info.cond_sep.append(e.topLeft[0]-master.topRight[0])
                if 'rbEnv' not in env_names:
                    info.cond_sep.append(0.0)
                    info.cond_sep.append(0.0)
                else:
                    for e in env_list:
                        if e.name == 'rbEnv':
                            info.cond_sep.append(e.botLeft[0]-c3_rbx)
                            info.cond_sep.append(e.topLeft[0]-c3_rtx)
        else:
            # 上层
            env_names = []
            for e in env_list:
                env_names.append(e.name)
            if 'ltEnv' in env_names:
                for e in env_list:
                    if e.name == 'ltEnv':
                        info.cond_sep.append(master.botLeft[0]-e.botRight[0])
                        info.cond_sep.append(master.topLeft[0]-e.topRight[0])
            else:
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
            if 'c4' not in env_names:      # 只要c2不在，rEnv一定不在
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
                info.cond_sep.append(0.0)
            else:
                for e in env_list:
                    if e.name == 'c4':
                        c4_rtx = e.topRight[0]
                        c4_rbx = e.botRight[0]
                        info.cond_sep.append(e.botLeft[0]-master.botRight[0])
                        info.cond_sep.append(e.topLeft[0]-master.topRight[0])
                if 'rtEnv' not in env_names:
                    info.cond_sep.append(0.0)
                    info.cond_sep.append(0.0)
                else:
                    for e in env_list:
                        if e.name == 'rtEnv':
                            info.cond_sep.append(e.botLeft[0]-c4_rbx)
                            info.cond_sep.append(e.topLeft[0]-c4_rtx)
    
    info.metal_H = master.botLeft[1]-bot_dict['botleft'].points[3][1]

    if T == 'PLATE3L':
        info.top_H = top_dict['topleft'].points[0][1]-bot_dict['botleft'].points[3][1]
        info.metal_top = top_dict['topleft'].points[0][1]-master.topLeft[1]
    
    #导体层厚度
    info.cond_thickness = master.height

    return info


def output(info, id, T, model_T):
    re_info = []

    if T == 'PLATE2L':
        re_info.append(info.boundary_lx)
        re_info.append(info.boundary_rx)
        for value in info.env_top_width.values():
            re_info.append(value)
        for value in info.env_bot_width.values(): 
            re_info.append(value)
        for value in info.env_K.values():
            re_info.append(value)
        re_info.append(info.master_top_width)
        re_info.append(info.master_bot_width)
        re_info.append(info.master_K)
        re_info.append(info.metal_H)
        re_info.append(info.cond_thickness)
        for sep in info.cond_sep:
            re_info.append(sep)
        for dw in info.damage_width:
            re_info.append(dw)
    elif T == 'PLATE3L':
        if model_T == 0:
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
        elif model_T == 1 or model_T == 3: 
            if id % 2 == 1:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.metal_H)
                re_info.append(info.metal_top)
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
        elif model_T == 2: 
            if id % 2 == 1:
                re_info.append(info.boundary_lx)
                re_info.append(info.boundary_rx)
                re_info.append(info.master_top_width)
                re_info.append(info.master_bot_width)
                re_info.append(info.master_K)
                re_info.append(info.cond_thickness/info.metal_H)
                re_info.append(info.cond_thickness/info.metal_top)
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
        if id % 2 == 1:
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            for value in info.env_top_width.values():
                re_info.append(value)
            for value in info.env_bot_width.values():
                re_info.append(value)
            for value in info.env_K.values():
                re_info.append(value)
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.master_K)
            re_info.append(info.metal_H)
            for sep in info.cond_sep:
                re_info.append(sep)
            for dw in info.damage_width:
                re_info.append(dw)
        else:
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.master_K)
            re_info.append(info.metal_H)

    return re_info
 

def parser(single,      #是否单个输入
           file1='hi',       #文件1路径
           file2='hi',       #文件2路径
           pre='hi',         #文件前缀
           post='hi',        #文件后缀
           input_num='hi',   #文件数量
           directory='hi'    #目录路径
           ):
    model_index = [[], [], [], []]
    if single:
        #先读一个文件
        master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern = read_txt_file(file1)
        #如果是PLATE2L，只需要读入一个文件
        if pattern == 'PLATE2L':
            #处理信息
            feature_matrix = []
            info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, 1)
            feature_list = output(info, 1, pattern, 0)
            feature_matrix.append(feature_list)
        #如果是PLATE3L或者STACK3L，需要读入连续的两个文件，第一个master为c1
        else:
            master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2 = read_txt_file(file2)
            assert pattern == pattern2, 'find different pattern in file1 and file2!'

            #如果master是c1(输入的第一个文件的master是c1)
            if master.name == 'c1':
                if master2.name == 'botcenter':
                    assert pattern == 'PLATE3L', 'file pattern error!Shoud be PLATE3L when one of the masters is botcenter!'
                    feature_matrix = [[], [], [], []]
                    # PLATE3L, 判断model type
                    if len(env_list) == 0:
                        modelT = model_type['c1']
                    elif len(env_list) == 3:
                        modelT = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list) == 1:
                        if env_list[0].name == 'c2':
                            modelT = model_type['c1_c2']
                        if env_list[0].name == 'lEnv':
                            modelT = model_type['c1_lEnv']
                    #得到输出
                    info1 = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, 1)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 2)
                    re1 = output(info1, 1, pattern, modelT)
                    re2 = output(info2, 2, pattern2, modelT)
                    re = re1+re2
                    feature_matrix[modelT].append(re)
                    model_index[modelT].append(0)
                elif master2.name == 'c2':
                    assert pattern == 'STACK3L', 'file pattern error!Shoud be STACK3L when one of the masters is c2'
                    feature_matrix = []
                    info1 = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, 1)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 2)
                    re1 = output(info1, 1, pattern, 0)
                    re2 = output(info2, 2, pattern, 0)
                    re = re1+re2
                    feature_matrix.append(re)
            #如果master2是c1(输入的第二个文件的master是c1)
            if master2.name == 'c1':
                if master.name == 'botcenter':
                    assert pattern2 == 'PLATE3L', 'file pattern error!Shoud be PLATE3L when one of the masters is botcenter!'
                    feature_matrix = [[], [], [], []]
                    # PLATE3L, 判断model type
                    if len(env_list2) == 0:
                        modelT = model_type['c1']
                    elif len(env_list2) == 3:
                        modelT = model_type['c1_c2_lEnv_rEnv']
                    elif len(env_list2) == 1:
                        if env_list2[0].name == 'c2':
                            modelT = model_type['c1_c2']
                        if env_list2[0].name == 'lEnv':
                            modelT = model_type['c1_lEnv']
                    #得到输出
                    info1 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 1) #info1是c1为master的文件输入信息
                    info2 = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, 2)
                    re1 = output(info1, 1, pattern2, modelT)
                    re2 = output(info2, 2, pattern, modelT)
                    re = re1+re2
                    feature_matrix[modelT].append(re)
                    model_index[modelT].append(i)
                elif master.name == 'c2':
                    assert pattern2 == 'STACK3L', 'file pattern error!Shoud be STACK3L when one of the masters is c2'
                    feature_matrix = []
                    info1 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 1)
                    info2 = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, 2)
                    re1 = output(info1, 1, pattern2, 0)
                    re2 = output(info2, 2, pattern, 0)
                    re = re1+re2
                    feature_matrix.append(re)
    else:
        #读入目录下的一组文件
        #先判断pattern
        file_path = directory+'/'+pre+'1'+post
        _, _, _, _, _, _, pattern = read_txt_file(file_path)
        if pattern == 'PLATE2L':
            feature_matrix = []
            for i in range(0, input_num):
                file_path = directory+'/'+pre+str(i+1)+post
                #read
                master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern_cur = read_txt_file(file_path)
                assert pattern_cur == pattern, 'find file pattern differs in this directory!'
                #提取信息
                info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern_cur, i+1)
                feature_list = output(info, i+1, pattern_cur, 0)
                feature_matrix.append(feature_list)
        elif pattern == 'PLATE3L':
            feature_matrix = [[], [], [], []]
            assert input_num%2==0, 'input number error!Shoud be even number!'
            sampling_num = int(input_num/2)

            for i in range(0, sampling_num):
                f1 = 2*i+1
                f2 = 2*i+2
                file1_path = directory+'/'+pre+str(f1)+post
                file2_path = directory+'/'+pre+str(f2)+post
                #read
                master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1 = read_txt_file(file1_path)
                master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2 = read_txt_file(file2_path)
                assert pattern1 == pattern2 and pattern1 == pattern, 'find different file patterns in this directory!'
                # 判断model type
                if len(env_list1) == 0:
                    modelT = model_type['c1']
                elif len(env_list1) == 3:
                    modelT = model_type['c1_c2_lEnv_rEnv']
                elif len(env_list1) == 1:
                    if env_list1[0].name == 'c2':
                        modelT = model_type['c1_c2']
                    if env_list1[0].name == 'lEnv':
                        modelT = model_type['c1_lEnv']
                #看下哪个master是c1，然后提取信息。info1一定是c1为master的那个文件的信息
                if master1.name == 'c1':
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1, 1)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 2)
                elif master2.name == 'c1':
                    info1 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 1)
                    info2 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1, 2)
                re1 = output(info1, 1, 'PLATE3L', modelT)
                re2 = output(info2, 2, 'PLATE3L', modelT)
                re = re1+re2
                feature_matrix[modelT].append(re)
                model_index[modelT].append(i)
        elif pattern == 'STACK3L':
            feature_matrix = []
            assert input_num%2==0, 'input number error!Shoud be even number!'
            sampling_num = int(input_num/2)

            for i in range(0, sampling_num):
                f1 = 2*i+1
                f2 = 2*i+2
                file1_path = directory+'/'+pre+str(f1)+post
                file2_path = directory+'/'+pre+str(f2)+post
                #read
                master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1 = read_txt_file(file1_path)
                master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2 = read_txt_file(file2_path)
                assert pattern1 == pattern2 and pattern1 == pattern, 'find different file patterns in this directory!'
                #看下哪个master是c1，然后提取信息。info1一定是c1为master的那个文件的信息
                if master1.name == 'c1':
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1, 1)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 2)
                elif master2.name == 'c1':
                    info1 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern2, 1)
                    info2 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern1, 2)
                re1 = output(info1, 1, 'STACK3L', 0)
                re2 = output(info2, 2, 'STACK3L', 0)
                re = re1+re2
                feature_matrix.append(re)

    return feature_matrix, model_index, pattern 




if __name__ == '__main__':
    start = time.time()
    feature_matrix, model_index, pattern = parser(single=False,
                               file1='/home/huadacup/HuadaJiutian/Cases/STACK3L/SUB-metal2-metal3_STACK3L/input/BEM_INPUT_1_126367.txt',
                               file2='/home/huadacup/HuadaJiutian/Cases/STACK3L/SUB-metal2-metal3_STACK3L/input/BEM_INPUT_2_126367.txt',
                               pre='BEM_INPUT_',
                               post='_133293.txt',
                               input_num=648,
                               directory='/home/huadacup/HuadaJiutian/Cases/PLATE3L/SUB-metal1-metal2_PLATE3L/input'
    )
    end = time.time()
    print(end-start)
    print(pattern)
    print(len(feature_matrix[0][0]))
    print(len(feature_matrix[1][0]))
    print(len(feature_matrix[2][0]))
    print(len(feature_matrix[3][0]))

