import readarrhythmia
import read_Sales_Transactions_Dataset_Weekly
import readsemeion

if __name__ == '__main__':
    semeion = readsemeion.ReadSemeion('semeion.data')
    arrhythmia = readarrhythmia.ReadArrhythmia('arrhythmia.data')
    sales_transactions_dataset = read_Sales_Transactions_Dataset_Weekly.ReadSalesTransactionsDatasetWeekly('Sales_Transactions_Dataset_Weekly.csv')
