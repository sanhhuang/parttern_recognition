import os

def ReadSeedsDataSet(arrhythmia_file_name):
    # open file
    #arrhythmia_file_name = "seeds_dataset.txt"
    origin_file = open(arrhythmia_file_name, 'r')

    # get all lines in file
    lines = origin_file.readlines()
    all_data = []
    labels = []
    for line in lines:
        # split as list
        vec = []
        s_num = ''
        for char in line:
            if (char >= '0' and char <= '9') or char == '.':
                s_num += char
            else:
                if len(s_num) > 0:
                    vec.append(float(s_num))
                    s_num = ''
        if len(s_num) > 0:
            vec.append(float(s_num))
            s_num = ''
        # print(vec)
        # add in list, double list
        all_data.append(vec[:-1])
        index = [0,1,3,4]
        #all_data.append([vec[0],vec[1]])
        labels.append(vec[-1])
    print('%s : %d' % (arrhythmia_file_name, len(all_data)))
    origin_file.close()
    return all_data, labels

if __name__ == "__main__":
    ReadSeedsDataSet('seeds_dataset.txt')
