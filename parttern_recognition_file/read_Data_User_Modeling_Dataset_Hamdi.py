import os
import xlrd

def ReadUserModelingDataSetHamdi(xls_file_name):
# open file
    #sheet = 'Data_User_Modeling_Dataset_Hamdi.xls'
    #读取xls文件,一定要把xlsx后缀改成xls
    xls_file = xlrd.open_workbook(xls_file_name)
    table= xls_file.sheet_by_name('Training_Data')
    nrows = table.nrows #总行数
    ncols = table.ncols #总列数
    limit = 0
    for limit in range(ncols):
        if(table.row_values(0)[limit] == "PEG"):
            break;
    all_data = []
    for i in range(1, nrows):
        all_data.append([float(table.row_values(i)[col]) for col in range(limit + 1)])
        print(all_data[-1])
    print('%s : %d' % (xls_file_name, len(all_data)))
    return all_data

if __name__ == '__main__':
    ReadUserModelingDataSetHamdi('Data_User_Modeling_Dataset_Hamdi.xls')
