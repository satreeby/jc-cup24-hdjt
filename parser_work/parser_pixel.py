import time


# /*方格的长、宽分别设为boundary长、宽的分数，这样保证输入维度一样*/
# /*需要面对导体可变化，即密度矩阵可以随时加入/删除导体*/
# /*只能处理梯形和矩形导体，平行四边形的damage还不一定能处理*/
# /*压线可能会有bug*/


class Master:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points    # points[0] is the coordinate of the bottomleft dot
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


class Bottom:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


class Top:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        self.width = points[1][0]-points[0][0]
        self.height = points[2][1]-points[1][1]
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


class Env:
    def __init__(self, name, num, points):
        self.name = name
        self.num = int(num)
        self.points = points
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


class Dielectric:
    def __init__(self, name, num, points, er):
        self.name = name
        self.num = int(num)
        self.points = points
        self.er = er
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


class Boundary_Polygon:
    def __init__(self, num, points):
        self.num = int(num)
        self.points = points
        # determine the four points at the four corners
        self.botLeft = []
        self.botRight = []
        self.topLeft = []
        self.topRight = []
        self.width = 0
        self.height = 0

    def setBoundary(self):
        topy = max(point[1] for point in self.points)
        boty = min(point[1] for point in self.points)
        top_points = [point for point in self.points if point[1]==topy]
        bot_points = [point for point in self.points if point[1]==boty]
        self.topLeft = min(top_points, key=lambda x:x[0])
        self.topRight = max(top_points, key=lambda x:x[0])
        self.botLeft = min(bot_points, key=lambda x:x[0])
        self.botRight = max(bot_points, key=lambda x:x[0])
         # calculate the width and height
        self.width = (self.botRight[0] - self.botLeft[0] + self.topRight[0] - self.topLeft[0]) / 2
        self.height = (self.topLeft[1] - self.botLeft[1] + self.topRight[1] - self.botRight[1]) / 2


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


def nonNegAndSetBound(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly):
    blx = -boundpoly.points[0][0]
    bby = -bot_dict['botleft'].points[0][1]

    #将所有坐标非负化
    for i in range(len(master.points)):
        master.points[i][0] += blx
        master.points[i][1] += bby
    for env in env_list:
        for i in range(len(env.points)):
            env.points[i][0] += blx
            env.points[i][1] += bby
    for bot in bot_dict.values():
        for i in range(len(bot.points)):
            bot.points[i][0] += blx
            bot.points[i][1] += bby
    for top in top_dict.values():
        for i in range(len(top.points)):
            top.points[i][0] += blx
            top.points[i][1] += bby
    for die in dielectric_list:
        for i in range(len(die.points)):
            die.points[i][0] += blx
            die.points[i][1] += bby
    for i in range(len(boundpoly.points)):
        boundpoly.points[i][0] += blx
        boundpoly.points[i][1] += bby
    #所有导体设置边界
    master.setBoundary()
    for env in env_list:
        env.setBoundary()
    for bot in bot_dict.values():
        bot.setBoundary()
    for top in top_dict.values():
        top.setBoundary()
    for die in dielectric_list:
        die.setBoundary()
    boundpoly.setBoundary()


def xOnline(topx, topy, botx, boty, y):
    '''return x on line'''
    return topx - (topy-y) * (topx - botx) / (topy - boty)


def yOnline(topx, topy, botx, boty, x):
    '''return y on line'''
    return topy - (topx - x) * (topy - boty) / (topx - botx)


def getDensity(divw, divh, r, master, env_list, bot_dict, top_dict, dielectric_list, boundpoly):
    # init density matrix and values
    density = []
    for i in range(divh):
        density.append([0.0 for j in range(divw)])
    ax = boundpoly.width/divw
    if len(top_dict) == 0:
        ay = boundpoly.points[-1][1]/divh
    else:
        ay = top_dict['topleft'].points[-1][1]/divh

    if len(top_dict) == 0:
        bty = boundpoly.topLeft[1]
    else:
        bty = top_dict['topleft'].topLeft[1]
    brx = boundpoly.botRight[0]

    #计算每个导体的density
    analyzeOneCOnd(master, ax, ay, density, bty, brx)
    for bot in bot_dict.values():
        analyzeOneCOnd(bot, ax, ay, density, bty, brx)
    if len(top_dict) != 0:
        for top in top_dict.values():
            analyzeOneCOnd(top, ax, ay, density, bty, brx)
    for env in env_list:
        analyzeOneCOnd(env, ax, ay, density, bty, brx)
    
    return density


def analyzeOneCOnd(cond, ax, ay, density, bty, brx):
    gridblx = int(cond.botLeft[0]/ax)
    gridbly = int(cond.botLeft[1]/ay)
    gridtlx = int(cond.topLeft[0]/ax)
    gridtly = int(cond.topLeft[1]/ay)
    gridtrx = int(cond.topRight[0]/ax)
    gridtry = int(cond.topRight[1]/ay)
    gridbrx = int(cond.botRight[0]/ax)
    gridbry = int(cond.botRight[1]/ay)
    # 压线修正

    if cond.topLeft[1] == bty:
        gridtly -= 1
    if cond.topRight[1] == bty:
        gridtry -= 1
    if cond.topRight[0] == brx:
        gridtrx -= 1
    if cond.botRight[0] == brx:
        gridbrx -= 1
    
    # 中间
    condlx_in = max(gridtlx, gridblx)
    condrx_in = min(gridtrx, gridbrx)
    if condrx_in-1 >= condlx_in+1:
        if gridtly-1 >= gridbly+1:
            #导体包含的部分
            for j in range(gridbly+1, gridtly):
                for i in range(condlx_in+1, condrx_in):
                    density[j][i] += 1
        for i in range(condlx_in+1, condrx_in):
            density[gridtly][i] += (cond.topLeft[1]-gridtly*ay)/ay
            density[gridbly][i] += ((gridbly+1)*ay-cond.botLeft[1])/ay
    #左边
    if cond.botLeft[0] == cond.topLeft[0]:
        for i in range(gridbly, gridtly+1):
            density[i][gridblx] += (ax-(cond.topLeft[0]-gridtlx*ax))/ax
    else:
        dotx = []
        dotx.append(cond.botLeft[0])
        for i in range(gridbly+1, gridtly+1):
            dotx.append(xOnline(cond.topLeft[0], cond.topLeft[1], cond.botLeft[0], cond.botLeft[1], i*ay))
        dotx.append(cond.topLeft[0])
        for i in range(gridbly, gridtly+1):
            relagridy = i-gridbly
            downgridx = int(dotx[relagridy]/ax)
            upgridx = int(dotx[relagridy+1]/ax)
            #斜线包含完整的格子，则格子在上下两个截点的右侧
            if(downgridx < condlx_in and upgridx < condlx_in):
                gridlx = max(downgridx, upgridx)
                for j in range(gridlx+1, condlx_in+1):
                    if i == gridbly:
                        density[i][j] += ((gridbly + 1) * ay - cond.botLeft[1]) / ay
                    elif i == gridtly:
                        density[i][j] += (cond.topLeft[1] - gridtly * ay) / ay
                    else:
                        density[i][j] += 1
            #斜线截出一个梯形，则只要考虑截点所在格子本身
            if downgridx == upgridx:
                if i == gridtly:
                    density[i][downgridx] += (((ax * (downgridx + 1) - dotx[relagridy]) + (ax * (downgridx + 1) - dotx[relagridy + 1])) / 2 / ax) * ((cond.topLeft[1] - i * ay) / ay)
                elif i == gridbly:
                    density[i][downgridx] += (((ax * (downgridx + 1) - dotx[relagridy]) + (ax * (downgridx + 1) - dotx[relagridy + 1])) / 2 / ax) * (((i + 1) * ay - cond.botLeft[1]) / ay)
                else:
                    density[i][downgridx] += ((ax*(downgridx+1)-dotx[relagridy]) + (ax * (downgridx + 1) - dotx[relagridy+1])) / 2 / ax
            #如果斜线截出三角形，在一条格线的两边
            if downgridx != upgridx:
                if cond.botLeft[0] > cond.topLeft[0]:
                    midy = yOnline(dotx[relagridy + 1], ay * (i + 1), dotx[relagridy], ay * i, ax * downgridx)
                    density[i][upgridx] += (downgridx * ax - dotx[relagridy + 1]) * (ay * (i + 1) - midy) / 2 / (ax * ay)
                    density[i][downgridx] += (ax * ay - (midy - i * ay) * (dotx[relagridy] - downgridx * ax) / 2) / (ax * ay)
                else:
                    midy = yOnline(dotx[relagridy + 1], ay * (i + 1), dotx[relagridy], ay * i, ax * upgridx)
                    density[i][downgridx] += (upgridx * ax - dotx[relagridy]) * (midy-i*ay) / 2 / (ax * ay)
                    density[i][upgridx] += (ax * ay - ((i+1)*ay-midy) * (dotx[relagridy+1] - upgridx * ax) / 2) / (ax * ay)
    #右边
    if cond.botRight[0] == cond.topRight[0]:
        for i in range(gridbry, gridtry+1):
            density[i][gridbrx] += (cond.topRight[0] - gridtrx * ax) / ax
    else:
        dotx = []
        dotx.append(cond.botRight[0])
        for i in range(gridbry+1, gridtry+1):
            dotx.append(xOnline(cond.topRight[0], cond.topRight[1], cond.botRight[0], cond.botRight[1], i*ay))
        dotx.append(cond.topRight[0])
        for i in range(gridbry, gridtry+1):
            relagridy = i-gridbry
            downgridx = int(dotx[relagridy]/ax)
            upgridx = int(dotx[relagridy+1]/ax)
            #斜线包含完整的格子，则格子在上下两个截点的右侧
            if downgridx > condrx_in and upgridx > condrx_in:
                gridrx = min(downgridx, upgridx)
                for j in range(condrx_in, gridrx):
                    if i == gridbry:
                        density[i][j] += ((gridbry + 1) * ay - cond.botRight[1]) / ay
                    elif i == gridtry:
                        density[i][j] += (cond.topRight[1] - gridtry * ay) / ay
                    else:
                        density[i][j] += 1
            #斜线截出一个梯形，则只要考虑截点所在格子本身
            if downgridx == upgridx:
                if i == gridtry:
                    density[i][downgridx] += (((dotx[relagridy] - downgridx * ax) + (dotx[relagridy + 1] - downgridx * ax)) / 2 / ax) * ((cond.topRight[1] - ay * i) / ay)
                elif i == gridbry:
                    density[i][downgridx] += (((dotx[relagridy] - downgridx * ax) + (dotx[relagridy + 1] - downgridx * ax)) / 2 / ax) * (((i + 1) * ay - cond.botRight[1]) / ay)
                else:
                    density[i][downgridx] += ((dotx[relagridy] - downgridx * ax) + (dotx[relagridy + 1] - downgridx * ax)) / 2 / ax
            #如果斜线截出三角形，在一条格线的两边
            if downgridx != upgridx:
                if cond.botRight[0] < cond.topRight[0]:
                    midy = yOnline(dotx[relagridy + 1], ay * (i + 1), dotx[relagridy], ay * i, ax * upgridx)
                    density[i][upgridx] += (dotx[relagridy + 1] - upgridx * ax) * (ay * (i + 1) - midy) / 2 / (ax * ay)
                    density[i][downgridx] += (ax * ay - (midy - i * ay) * (upgridx * ax - dotx[relagridy]) / 2) / (ax * ay)
                else:
                    midy = yOnline(dotx[relagridy + 1], ay * (i + 1), dotx[relagridy], ay * i, ax * downgridx)
                    density[i][downgridx] += ( dotx[relagridy]-downgridx*ax) * (midy - i * ay) / 2 / (ax * ay)
                    density[i][upgridx] += (ax * ay - ((i + 1) * ay - midy) * (downgridx * ax - dotx[relagridy + 1]) / 2) / (ax * ay)


def parser(patternPath,     #Cases directory path
           pattern,         #pattern
           metal,           #metal
           divw,            #grid number along x axis
           divh,            #grid number along y axis
           r                #round
           ):
    assert pattern == 'PLATE2L' or pattern == 'PLATE3L' or pattern == 'STACK3L', "pattern error!"
    density_list = []

    if pattern == 'PLATE2L':
        assert metal == '1' or metal == '2' or metal == '3', 'metal error!'

        input_num = 324
        for i in range(0, input_num):
            if metal == '1':
                file_path = patternPath+'\PLATE2L\SUB-metal1_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_131919.txt'
            if metal == '2':
                file_path = patternPath+'\PLATE2L\SUB-metal2_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_132273.txt'
            if metal == '3':
                file_path = patternPath+'\PLATE2L\SUB-metal3_PLATE2L\input\BEM_INPUT_'+str(i+1)+'_132771.txt'
            master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(file_path)
            nonNegAndSetBound(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly)
            density_list.append(getDensity(divw, divh, r, master, env_list, bot_dict, top_dict, dielectric_list, boundpoly))
    
    elif pattern == 'PLATE3L':
        assert metal == '12' or metal == '13' or metal == '23', 'metal error!'

        input_num = 648
        for i in range(0, input_num):
            if metal == '12':
                file_path = patternPath+'\PLATE3L\SUB-metal1-metal2_PLATE3L\input\BEM_INPUT_'+str(i+1)+'_133293.txt'
            if metal == '13':
                file_path = patternPath+'\PLATE3L\SUB-metal1-metal3_PLATE3L\input\BEM_INPUT_'+str(i+1)+'_138255.txt'
            if metal == '23':
                file_path = patternPath+'\PLATE3L\SUB-metal2-metal3_PLATE3L\input\BEM_INPUT_'+str(i+1)+'_133631.txt'
            master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(file_path)
            nonNegAndSetBound(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly)
            density_list.append(getDensity(divw, divh, r, master, env_list, bot_dict, top_dict, dielectric_list, boundpoly))
    
    elif pattern == 'STACK3L':
        assert metal == '12' or metal == '13' or metal == '23', 'metal error!'

        input_num = 288
        for i in range(0, input_num):
            if metal == '12':
                file_path = patternPath+'\STACK3L\SUB-metal1-metal2_STACK3L\input\BEM_INPUT_'+str(i+1)+'_129796.txt'
            if metal == '13':
                file_path = patternPath+'\STACK3L\SUB-metal1-metal3_STACK3L\input\BEM_INPUT_'+str(i+1)+'_131209.txt'
            if metal == '23':
                file_path = patternPath+'\STACK3L\SUB-metal2-metal3_STACK3L\input\BEM_INPUT_'+str(i+1)+'_126367.txt'
            master, env_list, bot_dict, top_dict, dielectric_list, boundpoly = read_txt_file(file_path)
            nonNegAndSetBound(master, env_list, bot_dict, top_dict, dielectric_list, boundpoly)
            density_list.append(getDensity(divw, divh, r, master, env_list, bot_dict, top_dict, dielectric_list, boundpoly))

    return density_list
    

if __name__ == '__main__':
    den_path = ".\parser_work\parser_output\density.txt"
    pattern_path = ".\Cases"
    divw = 230  
    divh = 200
    r = 4       

    # pattern3的divw建议不小于230，pattern1和2的divw建议不小于200，否则可能重叠

    #返回三维density_list，其中包含了每一个input的density矩阵，例如density_list[0]就是第一个input的密度矩阵
    start = time.time()
    density_list = parser(patternPath=pattern_path, #Cases文件夹路径
                        pattern='PLATE2L',
                        metal='3',
                        divw=divw,                #横向格子数量
                        divh=divh,                #纵向格子数量
                        r=r)                      #这个先不管
    end = time.time()
    print(end-start)
    import numpy as np
    array=np.array(density_list)
    print(array.shape)

    #打开den_path描述的density.txt文件，并把第一个input对应的密度矩阵输出到文件中作为例子查看
    l = len(density_list[0])            
    with open(den_path, 'w') as f:                  
        for i in range(l):
            for d in density_list[0][l-1-i]:
                print("{:.4f}".format(d), end=' ', file=f)
            print("", file=f)
        



