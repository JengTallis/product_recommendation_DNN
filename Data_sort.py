import pandas as pd

def  completeData (file):
    df = pd.read_csv('transformed.csv', nrows=1000000)
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, False])
    df.to_csv('sorted.csv',index=False, sep = ',', encoding = 'utf-8')

completeData("transformed.csv")