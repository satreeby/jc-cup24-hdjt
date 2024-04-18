import numpy as np
import math


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
        self.coor = []
        self.setcoor()
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
    
    def setcoor(self):
        self.coor.append(self.botLeft[0])
        self.coor.append(self.botLeft[1])
        self.coor.append(self.botRight[0])
        self.coor.append(self.botRight[1])
        self.coor.append(self.topRight[0])
        self.coor.append(self.topRight[1])
        self.coor.append(self.topLeft[0])
        self.coor.append(self.topLeft[1])
        self.coor.append(0.)


class Bottom:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]
        self.coor = []
        self.setcoor()
    
    def setcoor(self):
        for point in self.points:
            for p in point:
                self.coor.append(p)
        self.coor.append(0.)



class Top:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]
        self.coor = []
        self.setcoor()
    
    def setcoor(self):
        for point in self.points:
            for p in point:
                self.coor.append(p)
        self.coor.append(0.)


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
        self.coor = []
        self.setcoor()
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
    
    def setcoor(self):
        self.coor.append(self.botLeft[0])
        self.coor.append(self.botLeft[1])
        self.coor.append(self.botRight[0])
        self.coor.append(self.botRight[1])
        self.coor.append(self.topRight[0])
        self.coor.append(self.topRight[1])
        self.coor.append(self.topLeft[0])
        self.coor.append(self.topLeft[1])
        self.coor.append(0.)


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
        self.coor = []
        self.setcoor()
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
    
    def setcoor(self):
        self.coor.append(self.botLeft[0])
        self.coor.append(self.botLeft[1])
        self.coor.append(self.botRight[0])
        self.coor.append(self.botRight[1])
        self.coor.append(self.topRight[0])
        self.coor.append(self.topRight[1])
        self.coor.append(self.topLeft[0])
        self.coor.append(self.topLeft[1])
        self.coor.append(self.er)


class Boundary_Polygon:
    def __init__(self, num, points):
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]
        self.coor = []
        self.setcoor()
    
    def setcoor(self):
        for point in self.points:
            for p in point:
                self.coor.append(p)
        self.coor.append(0.)
        


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


def process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly):
    # info need to be returned
    info = []
    
    info.append(master.coor)
    for e in env_list:
        info.append(e.coor)
    for bot in bot_dict.values():
        info.append(bot.coor)
    # for top in top_dict.values():
        # info.append(top.coor)
    # for d in dielectric_list:
        # info.append(d.coor)

    return info


def parser(pattern,     
           metal, 
           pattern_path # pattern 所在文件夹路径
           ):
    feature_matrix = []
    assert pattern == 'PLATE2L' or pattern == 'PLATE3L' or pattern == 'STACK3L', 'pattern error!'

    if pattern == 'PLATE2L':
        assert metal == '1' or metal == '2' or metal == '3', 'metal error!'

        input_num = 324
        for i in range(0, input_num):
            if metal == '1':
                file_path = pattern_path+'\SUB-metal1_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_131919.txt'
            if metal == '2':
                file_path = pattern_path+'\SUB-metal2_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_132273.txt'
            if metal == '3':
                file_path = pattern_path+'\SUB-metal3_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_132771.txt'
            master, env_list, bot_dict, top_dict, dielectric_list, boundpoly, _ = read_txt_file(file_path)
            info = process_data(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly)
            if len(info) < 100:
                for i in range(100 - len(info)):
                    info.append([0 for j in range(9)])
            feature_matrix.append(info)
    
    elif pattern == 'PLATE3L':
        assert metal == '12' or metal == '13' or metal == '23', 'metal error!'

        input_num = 324
        for i in range(0, input_num):
            c1_id = 2*i+1
            botcenter_id = 2*i+2
            if metal == '12':
                file1_path = pattern_path+'\SUB-metal1-metal2_PLATE3L\input\BEM_INPUT_'+str(c1_id)+'_133293.txt'
                file2_path = pattern_path+'\SUB-metal1-metal2_PLATE3L\input\BEM_INPUT_'+str(botcenter_id)+'_133293.txt'
            if metal == '13':
                file1_path = pattern_path+'\SUB-metal1-metal3_PLATE3L\input\BEM_INPUT_'+str(c1_id)+'_138255.txt'
                file2_path = pattern_path+'\SUB-metal1-metal3_PLATE3L\input\BEM_INPUT_'+str(botcenter_id)+'_138255.txt'
            if metal == '23':
                file1_path = pattern_path+'\SUB-metal2-metal3_PLATE3L\input\BEM_INPUT_'+str(c1_id)+'_133631.txt'
                file2_path = pattern_path+'\SUB-metal2-metal3_PLATE3L\input\BEM_INPUT_'+str(botcenter_id)+'_133631.txt'
            master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
            master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
            info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1)
            info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2)
            feature_matrix.append(info1)
            feature_matrix.append(info2)
        
    elif pattern == 'STACK3L':
        assert metal == '12' or metal == '13' or metal =='23', 'metal error!'

        input_num = 144
        for i in range(0, input_num):
            c1_id = 2*i+1
            c2_id = 2*i+2
            if metal == '12':
                file1_path = pattern_path+'\SUB-metal1-metal2_STACK3L\input\BEM_INPUT_'+str(c1_id)+'_129796.txt'
                file2_path = pattern_path+'\SUB-metal1-metal2_STACK3L\input\BEM_INPUT_'+str(c2_id)+'_129796.txt'
            if metal == '13':
                file1_path = pattern_path+'\SUB-metal1-metal3_STACK3L\input\BEM_INPUT_'+str(c1_id)+'_131209.txt'
                file2_path = pattern_path+'\SUB-metal1-metal3_STACK3L\input\BEM_INPUT_'+str(c2_id)+'_131209.txt'
            if metal == '23':
                file1_path = pattern_path+'\SUB-metal2-metal3_STACK3L\input\BEM_INPUT_'+str(c1_id)+'_126367.txt'
                file2_path = pattern_path+'\SUB-metal2-metal3_STACK3L\input\BEM_INPUT_'+str(c2_id)+'_126367.txt'
            master1, env_list1, bot_dict1, top_dict1, dielectric_list1, boundpoly1, _ = read_txt_file(file1_path)
            master2, env_list2, bot_dict2, top_dict2, dielectric_list2, boundpoly2, _ = read_txt_file(file2_path)
            info1 = process_data(master1, env_list1, bot_dict1, top_dict1, dielectric_list1)
            info2 = process_data(master2, env_list2, bot_dict2, top_dict2, dielectric_list2)
            feature_matrix.append(info1)
            feature_matrix.append(info2)

    return feature_matrix




if __name__ == '__main__':
    feature_matrix = parser(pattern='PLATE2L', metal='1', pattern_path='.\Cases\PLATE2L')

    # Displaying the array
    array=np.array(feature_matrix)
    np.set_printoptions(threshold=1e6)
    file = open(".\\parser_work\\parser_output\\file1_coordinates.txt", "w+")
    content = str(array)
    file.write(content)
    file.close()

    nums_token=[]
    max_tokens=0
    for i in range(len(feature_matrix)):
        nums_token.append(len(feature_matrix[i]))
        if len(feature_matrix[i])>max_tokens:
            max_tokens=len(feature_matrix[i])
    print(nums_token)
    print(max_tokens)