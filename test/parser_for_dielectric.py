# -*- coding: gbk -*-

'''
ʳ�÷���������parser()����
��ȡͬһtype�����ļ���parser(type=1 or 2 or 3, Fpath= ��type_data�ļ���·������ʽΪ�ַ�����)
��ȡĳһ���ļ���parser(type=0, Fpath=���ļ�·������ʽΪ�ַ�����)
����ļ�output��Ϊ�����֣���һ����ÿһ����Ϣ������ţ���ʾ����ͬһtype�����ļ�����Щֵ����ͬ
                      �ڶ����ֵ�����û����ţ���ÿһ��input�ж���ͬ�����ÿһ��input���һ��ֵ
'''

'''
���裺
1. master��������(����Ҳ�����δ���)�����������µ׶���ƽ��
2. ���������Ǿ��λ������Σ�û����������
3. ���εĿ�����ϡ��±߳��ȵľ�ֵ��ʾ���߶������߶ȷ�Χ��ʾ
4.  ���ֻ��һ�㵼�壬��Ϊ��type1
    ������㵼�壬��master���ϲ㣬��Ϊ��type2
    ��������㵼�壬��master���²㣬��Ϊ��type3
    �жϷ�����������������½ǵ��������ֵ�������е����ȵ���Сֵ������Ϊ�������㵼��
5. �����Ϣʱ���������ݱ�����С�������λ
6. ÿһ��ĺ����ͬ
7. bottom��һ���Ǿ��Σ�����ʱ����ĸ�������
8. ����dielectric number��ʱ�������air_layer
'''


class Master:
    def __init__(self, name, num, points):
        #ֱ�ӻ��
        self.name = name
        self.num = int(num)
        self.points = points    #�����½ǿ�ʼ
        #�ڼ��������ʱ��������״�ĵ㶼�Ǵ����½ǿ�ʼ�����������
        self.seq_x = 0
        # ȷ���ĸ����ϵ��ĸ���
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # �������
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
        #ֱ�ӻ��
        self.name = name
        self.num = int(num)
        self.points = points    #�����½ǿ�ʼ
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]


class Env:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        # ��Ҫ����
        self.seq_x = 0
        # ȷ���ĸ����ϵ��ĸ���
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # �������
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
        # ȷ���ĸ����ϵ��ĸ���
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.setBoundary()
        # �������
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
        '''������Ҫ���ص���Ϣ'''
        self.master = None          # master
        self.env_list = None        # env�б�
        self.master_seq = 0         # master�ڵ�����λ��
        self.master_width = 0       # master���
        self.master_thickness = 0   # master���
        self.env_num = 0            # �����������
        self.env_seq = {}           # ��������λ���ֵ�
        self.env_width = {}         # �������ֵ�
        self.env_thickness = {}     # �������ֵ�
        self.cond_sep = []          # �������б�
        self.bot_l_space = 0        # layer��������ľ���
        self.bot_r_space = 0        # layer�Ҳ������ľ���
        self.bot_divider = 0        # layer���ҷָ���������
        self.layer_thickness = 0    # layer���
        self.cond_to_cond = 0       # ������߶Ȳ�
        self.cond_to_layer = 0      # ����ײ���layer�����ľ���
        self.boundary_width = 0     # �߽��
        self.boundary_height = 0    # �߽��
        # self.d_width = {}           # ���ʿ���ֵ�
        # self.d_thickness = {}       # ���ʺ���ֵ�
        # self.d_Er = {}              # ������Խ�糣���ֵ�
        self.die_info = {}          # ������Ϣ


class Info_All:
    def __init__(self):
        self.masterL = []       # master�б�
        self.env_listL = []     # env_list�б�
        self.master_seqL = []  # master�ڵ�����λ��
        self.master_widthL = []  # master���
        self.master_thicknessL = []  # master���
        self.env_numL = []  # �����������
        self.env_seqL = []  # ��������λ���ֵ�
        self.env_widthL = []  # �������ֵ�
        self.env_thicknessL = []  # �������ֵ�
        self.cond_sepL = []  # �������ֵ�
        self.bot_l_spaceL = []  # layer��������ľ���
        self.bot_r_spaceL = []  # layer�Ҳ������ľ���
        self.bot_dividerL = []  # layer���ҷָ���������
        self.layer_thicknessL = []  # layer���
        self.cond_to_condL = [] # ���������
        self.cond_to_layerL = []  # ����ײ���layer�����ľ���
        self.boundary_widthL = []  # �߽��
        self.boundary_heightL = []  # �߽��
        self.die_infoL = []     # ������Ϣ

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
                print("�ļ��������㣬�޷���ȡ��Ҫ��Ϣ��")
                return None
    except FileNotFoundError:
        print(f"�ļ� {file_path} δ�ҵ���")
        return None
    except Exception as e:
        print(f"��������: {str(e)}")
        return None


def process_data(master, env_list, bot_list, dielectric_list, boundpoly):
    # ��Ҫ���ص���Ϣ
    info = Info_One_input()

    # ���õ������½ǵ��������ж�type
    TYPE = 1# Ĭ����type1
    cond_layer_sep = 0# ��������㣬˳����㵼���֮��ľ���
    cond_to_bottom = master.points[0][1]-bot_list[0].points[3][1]# ���㵼��㵽bottom�ľ���(���master������)
    y_threshold = min([e.height for e in env_list])
    y_threshold = min(y_threshold, master.height)
    for e in env_list:
        if master.points[0][1]-e.points[0][1] > y_threshold:# master������
            TYPE=2
            cond_layer_sep = master.points[0][1]-e.topLeft[1]
            cond_to_bottom = e.points[0][1]-bot_list[0].points[3][1]
            break
        else:
            if e.points[0][1]-master.points[0][1]>y_threshold:# master������
                TYPE=3
                cond_layer_sep = e.points[0][1]-master.topLeft[1]
                break


    # ������Ų�˳��
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

    # ����separation
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
    sep_distance.append((sep_x[str(1) + 'lt']+sep_x[str(1)+'lb'])/2 - boundpoly.points[0][0])  # ����߽�ľ���
    sep_distance.append(boundpoly.points[1][0] - (sep_x[str(1 + len(env_list)) + 'rt']+sep_x[str(1 + len(env_list)) + 'rb'])/2)

    # layer���Ҷ�����Ŀ��
    cond_lx = 0  # ���������ߵĵ�ĺ�����
    cond_rx = 0  # ��������ұߵĵ�ĺ�����
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

    '''�Խ��������С�������λ'''
    # ���ؽ��
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
    for d in dielectric_list:       # ͬһ��die���ƿ��ܻ��ж�����
        infolist = [d.points[0][0], d.points[0][1], round(d.width, 3), round(d.height, 3), d.er]
        info.die_info[d.name].append(infolist)
    return info, TYPE


def output_info(INFO_ALL, input_num, TYPE):
    # ���ļ��Ա�д��
    with open('output.txt', 'w') as f:
        #��ӡ����ʿ��
        for i in range(input_num):
            for values in INFO_ALL.die_infoL[i].values():    # values��һ���б�
                for value in values:
                    for v in value:
                        print(v, end='  ', file=f)
                    print('', file=f)
            print('', file=f)


def parser(type=0, Fpath=''):
    assert type == 0 or type == 1 or type == 2 or type == 3, 'û������type!'

    INFO_ALL = Info_All()# ����input�ļ�����Ϣ
    if type == 0:
        master, env_list, bot_list, dielectric_list, boundpoly = read_txt_file(Fpath)
        info, t = process_data(master, env_list, bot_list, dielectric_list, boundpoly)
        INFO_ALL.collect(info)
        output_info(INFO_ALL, 1, t)
    else:
        #��ȡ�����ļ�
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


# ����
if __name__=="__main__":
    # ����ͬһtype�����ļ�
    parser(type=3, Fpath='./data')
    # ����һ���ļ�
    # parser(type=0, Fpath='../data/type2_data/BEM_INPUT_1_43817.txt')
    # parser(type=0, Fpath='../data/type1_data/BEM_INPUT_1_43652.txt')
    # parser(type=0, Fpath='../data/type3_data/BEM_INPUT_1_43924.txt')
