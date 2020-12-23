import os

def ReadSalesTransactionsDatasetWeekly(file_name):
# open file
    #file_name = 'Sales_Transactions_Dataset_Weekly.csv'
    origin_file = open(file_name, 'r')

    # get line in file
    line = origin_file.readline()
    vec = line.split(',' , -1)
    for i in range(len(vec)):
        if vec[i] == 'Normalized 0':
            break

    all_data = []
    while line:
        line = origin_file.readline()
        vec = line.split(',', -1)
        if len(vec[i:]) == 0:
            continue
        all_data.append([float(x) for x in vec[i:]])
    print('%s : %d' % (file_name, len(all_data)))
    origin_file.close()
    return all_data

if __name__ == '__main__':
    ReadSalesTransactionsDatasetWeekly('Sales_Transactions_Dataset_Weekly.csv')
