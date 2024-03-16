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
        self.master_width = 0       # master宽度
        self.boundary_width = 0     # 边界宽


class Info_All:
    def __init__(self):
        self.masterL = []       # master列表
        self.env_listL = []     # env_list列表
        self.master_widthL = []  # master宽度
        self.boundary_widthL = []  # 边界宽

    def collect(self, info):
        self.masterL.append(info.master)
        self.env_listL.append(info.env_list)
        self.master_widthL.append(info.master_width)
        self.boundary_widthL.append(info.boundary_width)


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
    y_threshold = min([e.height for e in env_list])
    y_threshold = min(y_threshold, master.height)
    for e in env_list:
        if master.points[0][1]-e.points[0][1] > y_threshold:# master在上面
            TYPE=2
            break
        else:
            if e.points[0][1]-master.points[0][1]>y_threshold:# master在下面
                TYPE=3
                break

    '''对结果保留到小数点后三位'''
    # 返回结果
    info.master = master
    info.env_list = env_list
    info.master_width = round(master.width, 3)
    info.boundary_width = round(boundpoly.width, 3)
    return info, TYPE


def output_info(INFO_ALL, input_num, TYPE):
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
    # parser(type=0, Fpath='../data/type2_data/BEM_INPUT_1_43817.txt')
    # parser(type=0, Fpath='../data/type1_data/BEM_INPUT_1_43652.txt')
    # parser(type=0, Fpath='../data/type3_data/BEM_INPUT_1_43924.txt')
