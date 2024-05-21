import math
import time

'''suppose:
1. all conductors are trapezoid, and the bottom layer are rectangles
2. the top and bottom surface of all trapezoids are flat, say points on these surface share the same y
3. 所有输入的第一个input的top、bottom、环境导体没有缺少，且top和bottom缺少的只有右边一个
4. topright和topcenter至少存在一个
6. 假设前后一组的两个文件的导体、电介质等重复内容相同
7. 所有导体的宽度都用上下底的长度分开表示
8. 同名damage的宽度都一样，且在导体的下方和左右都会出现，
damage的数量统一？
9. 对于缺失的环境导体，其对应宽度设为0
10. 层高用master的中心的y表示，如果有两层（pattern3）就把两层的高度都输出
11. STACK3L中只有上层有damage，且所有的metal在前后两个一组的文件中的位置、宽度等都一样
'''


CHECK_ISOSCELES_TRAPEZOID = False


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
        if CHECK_ISOSCELES_TRAPEZOID:
            self.WheIT()

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
    
    def WheIT(self):
        if abs(self.topLeft[0]-self.botLeft[0]) == abs(self.topRight[0] - self.botRight[0]):
            print(self.name + "是等腰梯形")
        else:
            print(self.name + "是等腰梯形")


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
        if CHECK_ISOSCELES_TRAPEZOID:
            self.WheIT()
    

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x: x[0])
        self.topRight = max(top_points, key=lambda x: x[0])
        self.botLeft = min(bot_points, key=lambda x: x[0])
        self.botRight = max(bot_points, key=lambda x: x[0])
    
    def WheIT(self):
        if abs(self.topLeft[0]-self.botLeft[0]) == abs(self.topRight[0] - self.botRight[0]):
            print(self.name + "是等腰梯形")
        else:
            print(self.name + "是等腰梯形")


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
        if CHECK_ISOSCELES_TRAPEZOID:
            self.WheIT()

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x: x[0])
        self.topRight = max(top_points, key=lambda x: x[0])
        self.botLeft = min(bot_points, key=lambda x: x[0])
        self.botRight = max(bot_points, key=lambda x: x[0])
    
    def WheIT(self):
        if abs(self.topLeft[0]-self.botLeft[0]) == abs(self.topRight[0] - self.botRight[0]):
            print(self.name + "是等腰梯形")
        else:
            print(self.name + "是等腰梯形")


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
        self.ave_bot_r_y = 0
        self.boundary_lx = 0     
        self.boundary_rx = 0
        self.ave_top_r_y = 0
        self.metal_H = 0
        self.cond_sep = []
        self.damage_width = []
        self.air_layer_thickness = 0
        self.metal_H_2 = 0  # STACK3L
        self.cond_sep_2 = []    # STACK3L
        self.cond_thickness = 0


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

                # 判断pattern
                if len(top_dict) != 0:
                    T = 'PLATE3L'
                else:
                    for e in env_list:
                        if e.botLeft[1] > master.topLeft[1] or e.topLeft[1] < master.botLeft[1]:
                            T = "STACK3L"
                            break
                    T = 'PLATE2L'

                return master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, T
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
    info.master_K = round((info.master_top_width-info.master_bot_width)/2/master.height, 6)
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
    if 4-len(info.damage_width) > 0:
        for i in range(4-len(info.damage_width)):
            info.damage_width.append(0)
    # 环境导体的处理需要分type
    if T == 'PLATE2L' or T == 'PLATE3L':
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
                WlEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WlEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
                KlEnv = round((WlEnv_top-WlEnv_bot)/2/e.height, 6)
            if e.name == 'c2':
                Wc2_top = round(e.topRight[0]-e.topLeft[0], 6)
                Wc2_bot = round(e.botRight[0]-e.botLeft[0], 6)
                Kc2 = round((Wc2_top-Wc2_bot)/2/e.height, 6)
            if e.name == 'rEnv':
                WrEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WrEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
                KrEnv = round((WrEnv_top-WrEnv_bot)/2/e.height, 6)
        info.env_top_width['lEnv'] = WlEnv_top
        info.env_bot_width['lEnv'] = WlEnv_bot
        info.env_K['lEnv'] = KlEnv
        info.env_top_width['c2'] = Wc2_top
        info.env_bot_width['c2'] = Wc2_bot
        info.env_K['c2'] = Kc2
        info.env_top_width['rEnv'] = WrEnv_top
        info.env_bot_width['rEnv'] = WrEnv_bot
        info.env_K['rEnv'] = KrEnv
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
        for e in env_list:
            if e.name == 'lbEnv':
                WlbEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WlbEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
            if e.name == 'c3':
                Wc3_top = round(e.topRight[0]-e.topLeft[0], 6)
                Wc3_bot = round(e.botRight[0]-e.botLeft[0], 6)
            if e.name == 'rbEnv':
                WrbEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WrbEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
            if e.name == 'ltEnv':
                WltEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WltEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
            if e.name == 'c4':
                Wc4_top = round(e.topRight[0]-e.topLeft[0], 6)
                Wc4_bot = round(e.botRight[0]-e.botLeft[0], 6)
            if e.name == 'rtEnv':
                WrtEnv_top = round(e.topRight[0]-e.topLeft[0], 6)
                WrtEnv_bot = round(e.botRight[0]-e.botLeft[0], 6)
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
    # 导体层间距
    if T == 'PLATE2L' or T == 'PLATE3L':
        env_names = []
        for e in env_list:
            env_names.append(e.name)
        if 'lEnv' in env_names:
            for e in env_list:
                if e.name == 'lEnv':
                    info.cond_sep.append(round(master.botLeft[0]-e.botRight[0], 6))
                    info.cond_sep.append(round(master.topLeft[0]-e.topRight[0], 6))
        else:
            info.cond_sep.append(0)
            info.cond_sep.append(0)
        if 'c2' not in env_names:      # 只要c2不在，rEnv一定不在
            info.cond_sep.append(0)
            info.cond_sep.append(0)
            info.cond_sep.append(0)
            info.cond_sep.append(0)
        else:
            for e in env_list:
                if e.name == 'c2':
                    c2_rtx = e.topRight[0]
                    c2_rbx = e.botRight[0]
                    info.cond_sep.append(round(e.botLeft[0]-master.botRight[0], 6))
                    info.cond_sep.append(round(e.topLeft[0]-master.topRight[0], 6))
            if 'rEnv' not in env_names:
                info.cond_sep.append(0)
                info.cond_sep.append(0)
            else:
                for e in env_list:
                    if e.name == 'rEnv':
                        info.cond_sep.append(round(e.botLeft[0]-c2_rbx, 6))
                        info.cond_sep.append(round(e.topLeft[0]-c2_rtx, 6))
    elif T == 'STACK3L':
        if id%2 == 1:
            # 下层
            env_names = []
            for e in env_list:
                env_names.append(e.name)
            master_lx = (master.topLeft[0]+master.botLeft[0])/2
            master_rx = (master.topRight[0]+master.botRight[0])/2
            if 'lbEnv' in env_names:
                for e in env_list:
                    if e.name == 'lbEnv':
                        e_lx = (e.topLeft[0]+e.botLeft[0])/2
                        e_rx = (e.topRight[0]+e.botRight[0])/2
                        # info.cond_sep.append(round(e_lx-boundpoly.points[0][0], 6))
                        info.cond_sep.append(round(master_lx-e_rx, 6))
            else:
                # info.cond_sep.append(round(master_lx-boundpoly.points[0][0], 6))
                info.cond_sep.append(4.5)
            if 'c3' not in env_names:      # 只要c2不在，rEnv一定不在
                info.cond_sep.append(4.5)
                info.cond_sep.append(4.5)
                # info.cond_sep.append(round(boundpoly.points[1][0]-master_rx, 6))
            else:
                for e in env_list:
                    if e.name == 'c3':
                        c3_lx = (e.topLeft[0]+e.botLeft[0])/2
                        c3_rx = (e.topRight[0]+e.botRight[0])/2
                        info.cond_sep.append(round(c3_lx-master_rx, 6))
                if 'rbEnv' not in env_names:
                    info.cond_sep.append(4.5)
                    # info.cond_sep.append(round(boundpoly.points[1][0]-c3_rx, 6))
                else:
                    for e in env_list:
                        if e.name == 'rbEnv':
                            rbEnv_lx = (e.topLeft[0]+e.botLeft[0])/2
                            rbEnv_rx = (e.topRight[0]+e.botRight[0])/2
                            info.cond_sep.append(round(rbEnv_lx-c3_rx, 6))
                            # info.cond_sep.append(round(boundpoly.points[1][0]-rbEnv_rx, 6))
            # 上层
            env_names = []
            for e in env_list:
                env_names.append(e.name)
            for e in env_list:
                if e.name == 'c2left':
                    c2left_lx = (e.topLeft[0]+e.botLeft[0])/2
                if e.name == 'c2':
                    c2_rx = (e.topRight[0]+e.botRight[0])/2
            if 'ltEnv' in env_names:
                for e in env_list:
                    if e.name == 'ltEnv':
                        e_lx = (e.topLeft[0]+e.botLeft[0])/2
                        e_rx = (e.topRight[0]+e.botRight[0])/2
                        # info.cond_sep_2.append(round(e_lx-boundpoly.points[0][0], 6))
                        info.cond_sep_2.append(round(c2left_lx-e_rx, 6))
            else:
                # info.cond_sep_2.append(round(c2left_lx-boundpoly.points[0][0], 6))
                info.cond_sep_2.append(4.5)
            if 'c4' not in env_names:      # 只要c2不在，rEnv一定不在
                info.cond_sep_2.append(4.5)
                info.cond_sep_2.append(4.5)
                # info.cond_sep_2.append(round(boundpoly.points[1][0]-c2_rx, 6))
            else:
                for e in env_list:
                    if e.name == 'c4':
                        c4_lx = (e.topLeft[0]+e.botLeft[0])/2
                        c4_rx = (e.topRight[0]+e.botRight[0])/2
                        info.cond_sep_2.append(round(c4_lx-c2_rx, 6))
                if 'rtEnv' not in env_names:
                    info.cond_sep_2.append(4.5)
                    # info.cond_sep_2.append(round(boundpoly.points[1][0]-c4_rx, 6))
                else:
                    for e in env_list:
                        if e.name == 'rtEnv':
                            rtEnv_lx = (e.topLeft[0]+e.botLeft[0])/2
                            rtEnv_rx = (e.topRight[0]+e.botRight[0])/2
                            info.cond_sep_2.append(round(rtEnv_lx-c4_rx, 6))
                            # info.cond_sep_2.append(round(boundpoly.points[1][0]-rtEnv_rx, 6))
    
    # metal_H需要分type
    if T == 'PLATE2L' or T == 'PLATE3L':
        info.metal_H = round(master.botLeft[1]-bot_dict['botleft'].points[3][1], 6)
    elif T == 'STACK3L':
        info.metal_H = round((master.topLeft[1]+master.botLeft[1])/2, 6)
        for e in env_list:
            if e.name == 'c2':
                info.metal_H_2 = round((e.topLeft[1]+e.botLeft[1])/2, 6)
                break

    if T == 'PLATE3L':
        if 'topright' in top_dict:
            temp = []
            for point in top_dict['topright'].points:
                # info.top_r_x.append(point[0])
                temp.append(point[1])
            info.ave_top_r_y = round(sum(temp)/len(temp), 6)
    
    #导体层厚度
    info.cond_thickness = master.height

    return info


def output(info, id, T):
    re_info = []

    if T == 'PLATE2L':
        re_info.append(info.boundary_lx)
        re_info.append(info.boundary_rx)
        for value in info.env_top_width.values():
            re_info.append(math.log(value+1))
        for value in info.env_bot_width.values(): 
            re_info.append(math.log(value+1))
        for value in info.env_K.values():
            re_info.append(value)
        re_info.append(math.log(info.master_top_width+1))
        re_info.append(math.log(info.master_bot_width+1))
        re_info.append(info.master_K)
        re_info.append(info.metal_H)
        re_info.append(info.cond_thickness)
        for sep in info.cond_sep:
            re_info.append(math.log(sep+1))
        for dw in info.damage_width:
            re_info.append(dw)
    elif T == 'PLATE3L':
        if id % 2 == 1:
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            for value in info.env_top_width.values():
                re_info.append(value)
            for value in info.env_bot_width.values():
                re_info.append(value)
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.metal_H)
            for sep in info.cond_sep:
                re_info.append(sep)
            for dw in info.damage_width:
                re_info.append(dw)
        else:
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.ave_top_r_y)
    elif T == 'STACK3L':
        if id % 2 == 1:
            re_info.append(info.boundary_lx)
            re_info.append(info.boundary_rx)
            for value in info.env_top_width.values():
                re_info.append(value)
            for value in info.env_bot_width.values():
                re_info.append(value)
            # re_info.append(info.env_num)
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)
            re_info.append(info.air_layer_thickness)
            re_info.append(info.metal_H)
            re_info.append(info.metal_H_2)
            for sep in info.cond_sep:
                re_info.append(sep)
            for sep in info.cond_sep_2:
                re_info.append(sep)
        else:
            re_info.append(info.master_top_width)
            re_info.append(info.master_bot_width)

    return re_info
 

def parser(pattern,     
           metal, 
           pattern_path, # pattern 所在文件夹路径
           gen, 
           input_num
           ):
    feature_matrix = []
    assert pattern == 'PLATE2L' or pattern == 'PLATE3L' or pattern == 'STACK3L', 'pattern error!'

    if gen:
        if pattern == 'PLATE2L':
            for i in range(0, input_num):
                file_path = pattern_path+"/file_"+str(i+1)+".txt"
                #read
                master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, _ = read_txt_file(file_path)
                #提取信息
                info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, pattern, i+1)
                feature_list = output(info, i+1, pattern)
                feature_matrix.append(feature_list)
        if pattern == 'PLATE3L':
            assert input_num%2==0, 'input number error!'
            input_num = input_num/2

            for i in range(0, input_num):
                c1_id = 2*i+1
                botcenter_id = 2*i+2
                file1_path = pattern_path+"/file_"+str(c1_id)+".txt"
                file2_path = pattern_path+"/file_"+str(botcenter_id)+".txt"
                #read
                master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
                master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
                #提取信息
                info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern, c1_id)
                info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern, botcenter_id)
                re1 = output(info1, c1_id, pattern)
                re2 = output(info2, botcenter_id, pattern)
                re = re1+re2
                feature_matrix.append(re)

            if pattern == 'STACK3L':
                assert input_num%2==0, 'input number error!'
                input_num = input_num/2

                for i in range(0, input_num):
                    c1_id = 2*i+1
                    c2_id = 2*i+2
                    file1_path = pattern_path+"/file_"+str(c1_id)+".txt"
                    file2_path = pattern_path+"/file_"+str(c2_id)+".txt"
                    #read
                    master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
                    master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
                    #提取信息
                    info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, pattern, c1_id)
                    info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, pattern, c2_id)
                    re1 = output(info1, c1_id, pattern)
                    re2 = output(info2, c2_id, pattern)
                    re = re1+re2
                    feature_matrix.append(re)
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
                master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, _ = read_txt_file(file_path)
                info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, 'PLATE2L', i+1)
                feature_matrix.append(output(info, i+1, 'PLATE2L'))
        
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
                master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
                master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
                info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, 'PLATE3L', c1_id)
                info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, 'PLATE3L', botcenter_id)
                re1 = output(info1, c1_id, 'PLATE3L')
                re2 = output(info2, botcenter_id, 'PLATE3L')
                re = re1+re2
                feature_matrix.append(re)
        
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
                re1 = output(info1, c1_id, 'STACK3L')
                re2 = output(info2, c2_id, 'STACK3L')
                re = re1+re2
                feature_matrix.append(re)

    return feature_matrix




if __name__ == '__main__':
    start = time.time()
    feature_matrix = parser(pattern='PLATE2L', metal='1', pattern_path='E:/FDUFiles/DasanX/College_IC_Competition/pythonProject/data/Cases', gen=False, input_num=0)

    #pattern='PLATE2L', metal = '1' or '2' or '3'
    #pattern='PLATE3L', metal = '12' or '13' or '23'
    #pattern='STACK3L', metal = '12' or '13' or '23'
    #pattern_path表示Cases文件夹所在路径
    end = time.time()
    print(end-start)


    print(feature_matrix)
    

