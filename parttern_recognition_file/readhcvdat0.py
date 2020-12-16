import os

def ReadHcvDat(file_name):
# open file
    #file_name = 'Sales_Transactions_Dataset_Weekly.csv'
    origin_file = open(file_name, 'r')

    # get line in file
    line = origin_file.readline()
    vec = line.split(',' , -1)
    for i in range(len(vec)):
        if vec[i] == 'ALB':
            break

    all_data = []
    while line:
        vec = line.split(',', -1)
        all_data.append(vec[i:])
        line = origin_file.readline()
    print('%s : %d' % (file_name, len(all_data)))
    origin_file.close()
    return all_data

if __name__ == '__main__':
    ReadHcvDat('hcvdat0.csv')
