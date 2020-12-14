import os

def ReadSemeion(file_name):
    # open file
    # file_name = "semeion.data"
    origin_file = open(file_name, 'r')

    # get all lines in file
    lines = origin_file.readlines()
    all_data = []
    for line in lines:
        # split as list
        vec = line.split(' ', -1)
        # add in list, double list
        all_data.append(vec)
    print('%s : %d' % (file_name, len(all_data)))
    origin_file.close()
    return all_data

if __name__ == '__main__' :
    ReadSemeion('semeion.data')
