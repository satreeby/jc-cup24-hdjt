class Result1:
    def __init__(self, w1=0.0, rightSpace=0.0, leftSpace=0.0, edgeSpace=0.0,
                  c12=0.0, c13=0.0, c1e=0.0, c1tr=0.0, c1tl=0.0, c1br=0.0, c1bl=0.0):
        self.w1 = w1
        self.rightSpace = rightSpace
        self.leftSpace = leftSpace
        self.edgeSpace = edgeSpace
        self.c12 = c12
        self.c13 = c13
        self.c1e = c1e
        self.c1tr = c1tr
        self.c1tl = c1tl
        self.c1br = c1br
        self.c1bl = c1bl

    def print_info(self):
        print(self.w1, self.rightSpace, self.leftSpace, self.edgeSpace, 
              self.c12, self.c13, self.c1e, self.c1tr, self.c1tl, self.c1br, self.c1bl)


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            next(file)
            result_list=[]
            for line in file:
                values=[float(num) for num in line.split()]
                result_instance=Result1(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10])
                result_list.append(result_instance)
            for result_i in result_list:
                result_i.print_info()
            return None
    except FileNotFoundError:
        print("no such file")
        return None
    except Exception as e:
        print("error")
        return None


# 调用函数，传入文件路径
file_path = 'C:\\project\\python\\huadajiutian\\type1.text'
read_txt_file(file_path)