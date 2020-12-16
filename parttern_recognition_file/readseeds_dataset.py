import os

def ReadSeedsDataSet(arrhythmia_file_name):
    # open file
    #arrhythmia_file_name = "seeds_dataset.txt"
    origin_file = open(arrhythmia_file_name, 'r')

    # get all lines in file
    lines = origin_file.readlines()
    all_data = []
    for line in lines:
        # split as list
        vec = line.split(',', -1)
        # add in list, double list
        all_data.append(vec[:-2])
    print('%s : %d' % (arrhythmia_file_name, len(all_data)))
    origin_file.close()
    return all_data

if __name__ == "__main__":
    ReadSeedsDataSet('seeds_dataset.txt')
