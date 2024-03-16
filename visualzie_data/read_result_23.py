class Result23:
    def __init__(self, mstwidth=0.0, space=0.0, digSpace=0.0, edgeSpace=0.0,
                  c12=0.0, c1e=0.0, c1t=0.0, c1b=0.0, c1d2=0.0):
        self.mstwidth = mstwidth
        self.space = space
        self.digSpace = digSpace
        self.edgeSpace = edgeSpace
        self.c12 = c12
        self.c1e = c1e
        self.c1t = c1t
        self.c1b = c1b
        self.c1d2 = c1d2

    def print_info(self):
        print('mstwidth:{:.4f}, space:{:.4f}, digSpace:{:.4f}, edgeSpace:{:.4f}, '
              'c12:{:.4f}, c1e:{:.4f}, c1t:{:.4f}, c1b:{:.4f}, c1d2:{:.4f}'.format(self.mstwidth, self.space,
              self.digSpace, self.edgeSpace,self.c12, self.c1e, self.c1t, self.c1b, self.c1d2))


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            next(file)
            result_list=[]
            for line in file:
                values = []
                for num in line.split():
                    if num != '|':
                        values.append(float(num))
                result_instance=Result23(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8])
                result_list.append(result_instance)
            for result_i in result_list:
                result_i.print_info()
            return None
    except FileNotFoundError:
        print("no such file")
        return None


# 调用函数，传入文件路径
file_path = './type2.txt'
read_txt_file(file_path)