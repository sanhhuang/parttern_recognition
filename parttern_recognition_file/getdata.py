import readhcvdat0
import readseeds_dataset
import read_Sales_Transactions_Dataset_Weekly

if __name__ == '__main__':
    hcvdat0 = readhcvdat0.ReadHcvDat('hcvdat0.csv')
    seeds_dataset = readseeds_dataset.ReadSeedsDataSet('seeds_dataset.txt')
    sales_transactions_dataset = read_Sales_Transactions_Dataset_Weekly.ReadSalesTransactionsDatasetWeekly('Sales_Transactions_Dataset_Weekly.csv')
