import pandas as pd
import csv

def  sortData (filein):
    df = pd.read_csv(filein)
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, False])
    df.to_csv('sorted.csv',index=False, sep = ',', encoding = 'utf-8')

def completeData(file):
    with open(file, 'r') as r, open("complete.csv", 'w', newline='') as wr:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')
        fn = next(inp)
        print (fn)
        out.writerow(fn)
        test = []
        for row in inp:
            if len(test) == 0:
                test.append(row)
            else:
                if row[1]== test[len(test)-1][1]:
                    test.append(row)
                else:
                    if len(test)==17:
                        for t in test:
                            out.writerow(t)
                    test = []
                    test.append(row)



sortData("transformed.csv")
completeData('sorted.csv')