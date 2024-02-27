# -*- coding: gbk -*-

'''
食用方法：调用parser()函数
读取同一type所有文件：parser(type=1 or 2 or 3, Fpath= 【type_data文件夹路径，格式为字符串】)
读取某一个文件：parser(type=0, Fpath=【文件路径，格式为字符串】)
输出文件output分为两部分，第一部分每一行信息都带序号，表示对于同一type所有文件，这些值都相同
                      第二部分的内容没有序号，在每一个input中都不同，针对每一个input输出一行值
'''

'''
假设：
1. master都是梯形(矩形也用梯形处理)，且梯形上下底都是平的
2. 环境导体是矩形或者梯形，没有其它可能
3. 梯形的宽度用上、下边长度的均值表示，高度用最大高度范围表示
4.  如果只有一层导体，认为是type1
    如果两层导体，且master在上层，认为是type2
    如果有两层导体，且master在下层，认为是type3
    判断方法：两个导体的左下角点纵坐标差值大于所有导体厚度的最小值，则视为存在两层导体
5. 输出信息时将所有数据保留到小数点后三位
6. 每一层的厚度相同
7. bottom层一定是矩形，用逆时针的四个点描述
8. 计算dielectric number的时候包括了air_layer
'''


class Master:
    def __init__(self, name, num, points):
        #直接获得
        self.name = name
        self.num = int(num)
        self.points = points    #从左下角开始
        #在计算排序的时候，所有形状的点都是从左下角开始，用这个点算
        self.seq_x = 0
        # 确定四个角上的四个点
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # 计算宽、高
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
        #直接获得
        self.name = name
        self.num = int(num)
        self.points = points    #从左下角开始
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]


class Env:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        # 需要计算
        self.seq_x = 0
        # 确定四个角上的四个点
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # 计算宽、高
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
        # 确定四个角上的四个点
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # 计算宽、高
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
        '''定义需要返回的信息'''
        self.master = None          # master
        self.env_list = None        # env列表
        self.master_seq = 0         # master在导体层的位置
        self.master_width = 0       # master宽度
        self.master_thickness = 0   # master厚度
        self.env_num = 0            # 环境导体个数
        self.env_seq = {}           # 环境导体位置字典
        self.env_width = {}         # 导体宽度字典
        self.env_thickness = {}     # 导体厚度字典
        self.cond_sep = []          # 导体间距列表
        self.bot_l_space = 0        # layer左侧多出来的距离
        self.bot_r_space = 0        # layer右侧多出来的距离
        self.bot_divider = 0        # layer左右分隔处横坐标
        self.layer_thickness = 0    # layer厚度
        self.cond_to_cond = 0       # 导体层间高度差
        self.cond_to_layer = 0      # 导体底部到layer顶部的距离
        self.boundary_width = 0     # 边界宽
        self.boundary_height = 0    # 边界高
        # self.d_width = {}           # 介质宽度字典
        # self.d_thickness = {}       # 介质厚度字典
        # self.d_Er = {}              # 介质相对介电常数字典
        self.die_info = {}          # 介质信息


class Info_All:
    def __init__(self):
        self.masterL = []       # master列表
        self.env_listL = []     # env_list列表
        self.master_seqL = []  # master在导体层的位置
        self.master_widthL = []  # master宽度
        self.master_thicknessL = []  # master厚度
        self.env_numL = []  # 环境导体个数
        self.env_seqL = []  # 环境导体位置字典
        self.env_widthL = []  # 导体宽度字典
        self.env_thicknessL = []  # 导体厚度字典
        self.cond_sepL = []  # 导体间距字典
        self.bot_l_spaceL = []  # layer左侧多出来的距离
        self.bot_r_spaceL = []  # layer右侧多出来的距离
        self.bot_dividerL = []  # layer左右分隔处横坐标
        self.layer_thicknessL = []  # layer厚度
        self.cond_to_condL = [] # 导体层间距离
        self.cond_to_layerL = []  # 导体底部到layer顶部的距离
        self.boundary_widthL = []  # 边界宽
        self.boundary_heightL = []  # 边界高
        self.die_infoL = []     # 介质信息

    def collect(self, info):
        self.masterL.append(info.master)
        self.env_listL.append(info.env_list)
        self.master_seqL.append(info.master_seq)
        self.master_widthL.append(info.master_width)
        self.master_thicknessL.append(info.master_thickness)
        self.env_numL.append(info.env_num)
        self.env_seqL.append(info.env_seq)
        self.env_widthL.append(info.env_width)
        self.env_thicknessL.append(info.env_thickness)
        self.cond_sepL.append(info.cond_sep)
        self.bot_l_spaceL.append(info.bot_l_space)
        self.bot_r_spaceL.append(info.bot_r_space)
        self.bot_dividerL.append(info.bot_divider)
        self.layer_thicknessL.append(info.layer_thickness)
        self.cond_to_condL.append(info.cond_to_cond)
        self.cond_to_layerL.append(info.cond_to_layer)
        self.boundary_widthL.append(info.boundary_width)
        self.boundary_heightL.append(info.boundary_height)
        self.die_infoL.append(info.die_info)


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

    # 利用导体左下角点纵坐标判断type
    TYPE = 1# 默认是type1
    cond_layer_sep = 0# 如果是两层，顺便计算导体层之间的距离
    cond_to_bottom = master.points[0][1]-bot_list[0].points[3][1]# 计算导体层到bottom的距离(如果master在下面)
    y_threshold = min([e.height for e in env_list])
    y_threshold = min(y_threshold, master.height)
    for e in env_list:
        if master.points[0][1]-e.points[0][1] > y_threshold:# master在上面
            TYPE=2
            cond_layer_sep = master.points[0][1]-e.topLeft[1]
            cond_to_bottom = e.points[0][1]-bot_list[0].points[3][1]
            break
        else:
            if e.points[0][1]-master.points[0][1]>y_threshold:# master在下面
                TYPE=3
                cond_layer_sep = e.points[0][1]-master.topLeft[1]
                break


    # 导体层排布顺序
    con_layer_seq = {}
    con_layer_seq[master.name] = (master.botLeft[0]+master.topLeft[0])/2
    for env_i in env_list:
        con_layer_seq[env_i.name] = (env_i.botLeft[0]+env_i.topLeft[0])/2
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
    sep_x[str(master.seq_x) + 'lt'] = master.topLeft[0]
    sep_x[str(master.seq_x) + 'lb'] = master.botLeft[0]
    sep_x[str(master.seq_x) + 'rt'] = master.topRight[0]
    sep_x[str(master.seq_x) + 'rb'] = master.botRight[0]
    for e in env_list:
        sep_x[str(e.seq_x) + 'lt'] = e.topLeft[0]
        sep_x[str(e.seq_x) + 'lb'] = e.botLeft[0]
        sep_x[str(e.seq_x) + 'rt'] = e.topRight[0]
        sep_x[str(e.seq_x) + 'rb'] = e.botRight[0]
    sep_distance.append((sep_x[str(1) + 'lt']+sep_x[str(1)+'lb'])/2 - boundpoly.points[0][0])  # 到左边界的距离
    sep_distance.append(boundpoly.points[1][0] - (sep_x[str(1 + len(env_list)) + 'rt']+sep_x[str(1 + len(env_list)) + 'rb'])/2)

    # layer左右多出来的宽度
    cond_lx = 0  # 导体层最左边的点的横坐标
    cond_rx = 0  # 导体层最右边的点的横坐标
    if master.seq_x == 1:
        cond_lx = (master.botLeft[0] + master.topLeft[0]) / 2
    else:
        if master.seq_x == 1 + len(env_list):
            cond_rx = (master.botRight[0] + master.topRight[0]) / 2
    for e in env_list:
        if e.seq_x == 1:
            cond_lx = (e.botLeft[0] + e.topLeft[0]) / 2
        else:
            if e.seq_x == 1 + len(env_list):
                cond_rx = (e.botRight[0] + e.topRight[0]) / 2

    '''对结果保留到小数点后三位'''
    # 返回结果
    info.master = master
    info.env_list = env_list
    info.master_seq = master.seq_x
    info.master_width = round(master.width, 3)
    info.master_thickness = round(master.height, 3)
    info.env_num = len(env_list)
    for e in env_list:
        info.env_seq[e.name] = e.seq_x
        info.env_width[e.name] = round(e.width, 3)
        info.env_thickness[e.name] = round(e.height, 3)
    for s in range(len(sep_distance)):
        sep_distance[s] = round(sep_distance[s], 3)
    info.cond_sep = sep_distance
    info.bot_l_space = round(cond_lx - bot_list[0].points[0][0], 3)
    info.bot_r_space = round(bot_list[1].points[1][0] - cond_rx, 3)
    info.bot_divider = bot_list[0].points[1][0]
    info.layer_thickness = round(bot_list[0].height, 3)
    info.cond_to_cond = round(cond_layer_sep, 3)
    info.cond_to_layer = round(cond_to_bottom, 3)
    info.boundary_width = round(boundpoly.width, 3)
    info.boundary_height = round(boundpoly.height, 3)
    for d in dielectric_list:
        info.die_info[d.name] = []
    for d in dielectric_list:       # 同一个die名称可能会有多个情况
        infolist = [d.points[0][0], d.points[0][1], round(d.width, 3), round(d.height, 3), d.er]
        info.die_info[d.name].append(infolist)
    return info, TYPE


def output_info(INFO_ALL, input_num, TYPE):
    # 打开文件以便写入
    with open('output.txt', 'w') as f:
        #打印电介质宽度
        for i in range(input_num):
            for values in INFO_ALL.die_infoL[i].values():    # values是一个列表
                for value in values:
                    for v in value:
                        print(v, end='  ', file=f)
                    print('', file=f)
            print('', file=f)


def parser(type=0, Fpath=''):
    assert type == 0 or type == 1 or type == 2 or type == 3, '没有这种type!'

    INFO_ALL = Info_All()# 所有input文件的信息
    if type == 0:
        master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(Fpath)
        info, t = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
        INFO_ALL.collect(info)
        output_info(INFO_ALL, 1, t)
    else:
        #读取所有文件
        if type == 1:
            input_num = 64
            for i in range(0, input_num):
                file_path = Fpath+'/type1_data/BEM_INPUT_' + str(i + 1) + '_43652.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            output_info(INFO_ALL, input_num, 1)
        if type == 2:
            input_num = 48
            for i in range(0, input_num):
                file_path = Fpath+'/type2_data/BEM_INPUT_' + str(i + 1) + '_43817.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            output_info(INFO_ALL, input_num, 2)
        if type == 3:
            input_num = 32
            for i in range(0, input_num):
                file_path = Fpath+'/type3_data/BEM_INPUT_' + str(i + 1) + '_43924.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            output_info(INFO_ALL, input_num, 3)


# 调用
if __name__=="__main__":
    # 读入同一type所有文件
    parser(type=3, Fpath='./data')
    # 读入一个文件
    # parser(type=0, Fpath='../data/type2_data/BEM_INPUT_1_43817.txt')
    # parser(type=0, Fpath='../data/type1_data/BEM_INPUT_1_43652.txt')
    # parser(type=0, Fpath='../data/type3_data/BEM_INPUT_1_43924.txt')
