import data_utils as DTool
import  numpy as np

# str = "D:\\Data\\CuttingFile"

def loading_content(path):
    f = open(path)
    return f.readlines()

def process_file_content(file_content, path):
    res = np.zeros((6, len(file_content) , 23 , 1), float)
    for i_frame in range(139,len(file_content)):
        vector = file_content[i_frame].split()
        vector = [float(i) for i in vector]
        i_chanel = 0
        i_node = 0
        for i in range(len(vector)):
            res[i_chanel][i_frame-139][i_node][0] = vector[i]
            i_chanel += 1
            i_chanel %= 6
            if (i % 6 == 5):
                i_node += 1

    return res

def loading_data (path):
    dataset = DTool.get_dataset(path)
    path = DTool.get_motion_paths_and_labels(dataset)
    X = []
    Y = []

    for index in range(len(path[0])):
        label = path[1][index]
        file_content = loading_content(path[0][index])
        x_min = process_file_content(file_content, path[0][index])
        X.append(x_min)
        Y.append(label)

    return (X , Y)

# Note : Chỉ cần dùng method loading_data(đường dẫn)

# (X,Y) = loading_data(str)