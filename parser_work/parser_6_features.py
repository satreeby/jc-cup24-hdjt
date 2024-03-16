<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
# -*- coding: gbk -*-

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
9. 统一个pattern如有dielectric重复，其重复次数在不同采样点中必一样
'''



=======
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py
class Master:
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
<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
        '''定义需要返回的信息'''
        self.master = None          # master
        self.env_list = None        # env列表
        self.master_width = 0       # master宽度
        self.boundary_width = 0     # 边界宽
=======

        self.master = None
        self.env_list = None
        self.master_seq = 0
        self.master_width = 0
        self.master_thickness = 0
        self.env_num = 0
        self.env_seq = {}
        self.env_width = {}
        self.env_thickness = {}
        self.cond_sep = []
        self.bot_l_space = 0
        self.bot_r_space = 0
        self.bot_divider = 0
        self.layer_thickness = 0
        self.cond_to_cond = 0
        self.cond_to_layer = 0
        self.boundary_width = 0
        self.boundary_height = 0
        # self.d_width = {}
        # self.d_thickness = {}
        # self.d_Er = {}
        self.die_info = {}
        self.boundary_leftx = 0
        self.boundary_rightx = 0
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py


class Info_All:
    def __init__(self):
<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
        self.masterL = []       # master列表
        self.env_listL = []     # env_list列表
        self.master_widthL = []  # master宽度
        self.boundary_widthL = []  # 边界宽
=======
        self.masterL = []
        self.env_listL = []
        self.master_seqL = []
        self.master_widthL = []
        self.master_thicknessL = []
        self.env_numL = []
        self.env_seqL = []
        self.env_widthL = []
        self.env_thicknessL = []
        self.cond_sepL = []
        self.bot_l_spaceL = []
        self.bot_r_spaceL = []
        self.bot_dividerL = []
        self.layer_thicknessL = []
        self.cond_to_condL = []
        self.cond_to_layerL = []
        self.boundary_widthL = []
        self.boundary_heightL = []
        self.die_infoL = []
        self.boundary_leftxL = []
        self.boundary_rightxL = []
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py

    def collect(self, info):
        self.masterL.append(info.master)
        self.env_listL.append(info.env_list)
        self.master_widthL.append(info.master_width)
        self.boundary_widthL.append(info.boundary_width)
<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
=======
        self.boundary_heightL.append(info.boundary_height)
        self.die_infoL.append(info.die_info)
        self.boundary_leftxL.append(info.boundary_leftx)
        self.boundary_rightxL.append(info.boundary_rightx)
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py


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
                return None
    except FileNotFoundError:
        print(f"file {file_path} not found")
        return None
    except Exception as e:
        print(f"error: {str(e)}")
        return None


def process_data(master, env_list, bot_list, dielectric_list, boundpoly):

    info = Info_One_input()

<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
    # 利用导体左下角点纵坐标判断type
    TYPE = 1# 默认是type1
=======
    TYPE = 1
    cond_layer_sep = 0
    cond_to_bottom = master.points[0][1]-bot_list[0].points[3][1]
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py
    y_threshold = min([e.height for e in env_list])
    y_threshold = min(y_threshold, master.height)
    for e in env_list:
        if master.points[0][1]-e.points[0][1] > y_threshold:
            TYPE=2
            break
        else:
            if e.points[0][1]-master.points[0][1]>y_threshold:
                TYPE=3
                break

<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
    '''对结果保留到小数点后三位'''
    # 返回结果
=======

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
    sep_distance.append((sep_x[str(1) + 'lt']+sep_x[str(1)+'lb'])/2 - boundpoly.points[0][0])
    sep_distance.append(boundpoly.points[1][0] - (sep_x[str(1 + len(env_list)) + 'rt']+sep_x[str(1 + len(env_list)) + 'rb'])/2)

    cond_lx = 0
    cond_rx = 0
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

>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py
    info.master = master
    info.env_list = env_list
    info.master_width = round(master.width, 3)
    info.boundary_width = round(boundpoly.width, 3)
<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
=======
    info.boundary_height = round(boundpoly.height, 3)
    for d in dielectric_list:
        info.die_info[d.name] = []
    for d in dielectric_list:
        infolist = [d.points[0][0], d.points[0][1], round(d.width, 3), round(d.height, 3), d.er]
        info.die_info[d.name].append(infolist)
    info.boundary_leftx = boundpoly.points[0][0]
    info.boundary_rightx = boundpoly.points[1][0]
>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py
    return info, TYPE


def output_info(INFO_ALL, input_num, TYPE):
<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
    # 打开文件以便写入
    with open('output.txt', 'w') as f:
        print('type'+str(TYPE), file=f)
        if TYPE == 1:
            print('w1\t\trightspace\tleftspace\tedgespace\tWb', file=f)
            for i in range(input_num):
                rightspace = 0
                leftspace = 0
                edgespace = 0
                e_c2 = None
                for e in INFO_ALL.env_listL[i]:
                    if e.name == 'c2':
                        e_c2 = e
                        rightspace = round(abs(e.topLeft[0] - INFO_ALL.masterL[i].topRight[0] + e.botLeft[0] -
                                          INFO_ALL.masterL[i].botRight[0]) / 2, 3)
                    if e.name == 'c3':
                        leftspace = round(abs(
                            INFO_ALL.masterL[i].topLeft[0] - e.topRight[0] + INFO_ALL.masterL[i].botLeft[0] -
                            e.botRight[0]) / 2, 3)
                    if e.name == 'c2e':
                        edgespace = round(abs(e.topLeft[0] - e_c2.topRight[0] + e.botLeft[0] - e_c2.botRight[0]) / 2, 3)
                print('{:.3f}\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(INFO_ALL.master_widthL[i],
                                                                                      rightspace,
                                                                                      leftspace,
                                                                                      edgespace,
                                                                                      INFO_ALL.boundary_widthL[i]), file=f)


        if TYPE == 2:
            print('mstwidth\tspace\tdigSpace\tedgespace\t\tWb', file=f)
            for i in range(input_num):
                space = 0
                digSpace = 0
                edgespace = 0
                for e in INFO_ALL.env_listL[i]:
                    if e.name == 'c2':
                        space = round(abs(e.topLeft[0]-INFO_ALL.masterL[i].topRight[0]+e.botLeft[0]-INFO_ALL.masterL[i].botRight[0])/2, 3)
                    if e.name == 'd2':
                        digSpace = round(abs(e.topLeft[0]-INFO_ALL.masterL[i].topRight[0]+e.botLeft[0]-INFO_ALL.masterL[i].botRight[0])/2, 3)
                    if e.name == 'c1Env':
                        edgespace = round(abs(INFO_ALL.masterL[i].topLeft[0]-e.topRight[0]+INFO_ALL.masterL[i].botLeft[0]-e.botRight[0])/2, 3)
                print('{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(INFO_ALL.master_widthL[i],
                                                                                      space,
                                                                                      digSpace,
                                                                                      edgespace,
                                                                                      INFO_ALL.boundary_widthL[i]), file=f)

        if TYPE == 3:
            print('mstwidth\tspace\tdigSpace\tedgespace\t\tWb', file=f)
            for i in range(input_num):
                space = 0
                digSpace = 0
                edgespace = 0
                for e in INFO_ALL.env_listL[i]:
                    if e.name == 'c2':
                        space = round(abs(e.topLeft[0]-INFO_ALL.masterL[i].topRight[0]+e.botLeft[0]-INFO_ALL.masterL[i].botRight[0])/2, 3)
                    if e.name == 'd2':
                        digSpace = round(abs(e.topLeft[0]-INFO_ALL.masterL[i].topRight[0]+e.botLeft[0]-INFO_ALL.masterL[i].botRight[0])/2, 3)
                    if e.name == 'c1Env':
                        edgespace = round(abs(INFO_ALL.masterL[i].topLeft[0]-e.topRight[0]+INFO_ALL.masterL[i].botLeft[0]-e.botRight[0])/2, 3)
                print('{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(
                    INFO_ALL.master_widthL[i],
                    space,
                    digSpace,
                    edgespace,
                    INFO_ALL.boundary_widthL[i],),file=f)
=======
    re_info = []

    if TYPE == 1:
        for i in range(input_num):
            rightspace = 0
            leftspace = 0
            edgespace = 0
            e_c2 = None
            for e in INFO_ALL.env_listL[i]:
                if e.name == 'c2':
                    e_c2 = e
                    rightspace = round(abs(e.topLeft[0] - INFO_ALL.masterL[i].topRight[0] + e.botLeft[0] -
                                        INFO_ALL.masterL[i].botRight[0]) / 2, 3)
                if e.name == 'c3':
                    leftspace = round(abs(
                        INFO_ALL.masterL[i].topLeft[0] - e.topRight[0] + INFO_ALL.masterL[i].botLeft[0] -
                        e.botRight[0]) / 2, 3)
                if e.name == 'c2e':
                    edgespace = round(abs(e.topLeft[0] - e_c2.topRight[0] + e.botLeft[0] - e_c2.botRight[0]) / 2, 3)
            temp_info = []
            temp_info.append(INFO_ALL.master_widthL[i])
            temp_info.append(rightspace)
            temp_info.append(leftspace)
            temp_info.append(edgespace)
            temp_info.append(INFO_ALL.boundary_leftxL[i])
            temp_info.append(INFO_ALL.boundary_rightxL[i])
            re_info.append(temp_info)

    if TYPE == 2 or TYPE == 3:
        for i in range(input_num):
            space = 0
            digSpace = 0
            edgespace = 0
            for e in INFO_ALL.env_listL[i]:
                if e.name == 'c2':
                    space = round(abs(e.topLeft[0] - INFO_ALL.masterL[i].topRight[0] + e.botLeft[0] -
                                      INFO_ALL.masterL[i].botRight[0]) / 2, 3)
                if e.name == 'd2':
                    digSpace = round(abs(e.topLeft[0] - INFO_ALL.masterL[i].topRight[0] + e.botLeft[0] -
                                         INFO_ALL.masterL[i].botRight[0]) / 2, 3)
                if e.name == 'c1Env':
                    edgespace = round(abs(
                        INFO_ALL.masterL[i].topLeft[0] - e.topRight[0] + INFO_ALL.masterL[i].botLeft[0] -
                        e.botRight[0]) / 2, 3)
            temp_info = []
            temp_info.append(INFO_ALL.master_widthL[i])
            temp_info.append(space)
            temp_info.append(digSpace)
            temp_info.append(edgespace)
            temp_info.append(INFO_ALL.boundary_leftxL[i])
            temp_info.append(INFO_ALL.boundary_rightxL[i])
            re_info.append(temp_info)

    return re_info


>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py


def parser(type=0, Fpath=''):
    assert type == 0 or type == 1 or type == 2 or type == 3, 'no such type!'

    INFO_ALL = Info_All()
    if type == 0:
        master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(Fpath)
        info, t = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
        INFO_ALL.collect(info)
        return output_info(INFO_ALL, 1, t)
    else:
        if type == 1:
            input_num = 64
            for i in range(0, input_num):
                file_path = Fpath+'/type1_data/BEM_INPUT_' + str(i + 1) + '_43652.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            return output_info(INFO_ALL, input_num, 1)
        if type == 2:
            input_num = 48
            for i in range(0, input_num):
                file_path = Fpath+'/type2_data/BEM_INPUT_' + str(i + 1) + '_43817.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            return output_info(INFO_ALL, input_num, 2)
        if type == 3:
            input_num = 32
            for i in range(0, input_num):
                file_path = Fpath+'/type3_data/BEM_INPUT_' + str(i + 1) + '_43924.txt'
                master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(file_path)
                info, _ = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
                INFO_ALL.collect(info)
            return output_info(INFO_ALL, input_num, 3)



<<<<<<< HEAD:visualzie_data/parser_for_dielectric.py
'''
食用方法：调用parser()函数
读取同一type所有文件：parser(type=1 or 2 or 3, Fpath= 【type_data文件夹路径，格式为字符串】)
读取某一个文件：parser(type=0, Fpath=【文件路径，格式为字符串】)
输出文件为output
'''
# 调用
if __name__=="__main__":
    # 读入同一type所有文件
    parser(type=1, Fpath='./data')
    # 读入一个文件
=======
if __name__=="__main__":

    print(parser(type=3, Fpath='../data'))

>>>>>>> 1229e571233c999dd50307694336beb30711065a:parser_work/parser_6_features.py
    # parser(type=0, Fpath='../data/type2_data/BEM_INPUT_1_43817.txt')
    # parser(type=0, Fpath='../data/type1_data/BEM_INPUT_1_43652.txt')
    # parser(type=0, Fpath='../data/type3_data/BEM_INPUT_1_43924.txt')
