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

        self.xcoors = []
        self.boundary_leftx = 0
        self.boundary_rightx = 0


class Info_All:
    def __init__(self):
        self.xcoorsL = []
        self.boundary_leftxL = []
        self.boundary_rightxL = []

    def collect(self, info):
        self.xcoorsL.append(info.xcoors)
        self.boundary_leftxL.append(info.boundary_leftx)
        self.boundary_rightxL.append(info.boundary_rightx)


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
        return None
    except Exception as e:
        return None


def process_data(master, env_list, bot_list, dielectric_list, boundpoly):

    info = Info_One_input()


    TYPE = 1
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


    # find the dielectrics that stay at the same layer with conductors
    for d in dielectric_list:
        if d.botLeft[1] > master.botLeft[1]-0.001 and d.topLeft[1] < master.topLeft[1]+0.001:
            if d.points[0][0] not in info.xcoors:
                info.xcoors.append(d.points[0][0])
        else:
            for e in env_list:
                if d.botLeft[1] > e.botLeft[1]-0.001 and d.topLeft[1] < e.topLeft[1]+0.001:
                    if d.points[0][0] not in info.xcoors:
                        info.xcoors.append(d.points[0][0])
                    break

    info.xcoors.append(master.points[0][0])
    for en in env_list:
        info.xcoors.append(en.points[0][0])

    info.boundary_leftx = round(boundpoly.points[0][0], 3)
    info.boundary_rightx = round(boundpoly.points[1][0], 3)
    return info, TYPE


def output_info(INFO_ALL, input_num, TYPE):
    re_info = []
    with open('output.txt', 'w') as f:
        print(TYPE)
        for i in range(input_num):
            info_list = []
            for x in INFO_ALL.xcoorsL[i]:
                info_list.append(x)
            info_list.append(INFO_ALL.boundary_leftxL[i])
            info_list.append(INFO_ALL.boundary_rightxL[i])
            re_info.append(info_list)
    return re_info



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



if __name__=="__main__":

    print(parser(type=1, Fpath='./data'))

    # parser(type=0, Fpath='./data/type2_data/BEM_INPUT_1_43817.txt')
    # parser(type=0, Fpath='./data/type1_data/BEM_INPUT_1_43652.txt')
    # parser(type=0, Fpath='./data/type3_data/BEM_INPUT_1_43924.txt')
