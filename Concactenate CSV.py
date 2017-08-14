import csv
import pandas as pd

def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1
    else:
        return list.index(input)

def changeDate (row):
    m = ['16463', '16494', '16522', '16553', '16583', '16614', '16644', '16675', '16706', '16736', '16767', '16797', '16828', '16859', '16888', '16919', '16949']
    if checkList(m, row[0]) == -1:
        row[0] = int(float(row[0]))
        row[0] +=1
        row[0] = str(row[0])
    else:
        row[0] = int(float(row[0]))
        row[0] -=1
        row[0] = str(row[0])
    return row


def concac (fileA, fileB):
    with open(fileA, 'r') as r1, open(fileB, 'r') as r2, open('compile/predicted.csv', 'w', newline='') as wr:
        out = csv.writer(wr, delimiter=",", quotechar='|')
        inp1 = csv.reader (r1, delimiter=",", quotechar='|')
        inp2 = csv.reader(r2, delimiter=",", quotechar='|')
        count = 0
        fn = next(inp1)
        out.writerow(fn)

        for row in inp1:
            row = changeDate(row)
            out.writerow(row)
            count+=1


        fn = next(inp2)
        for row in inp2:
            out.writerow(row)
            count+=1

        print (count)
        print(count%17)

def  sortData (filein):
    df = pd.read_csv(filein)
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, True])
    df.to_csv('sorted_predicted.csv',index=False, sep = ',', encoding = 'utf-8')

def countRow(file):
    with open(file, 'r') as r:
        inp = csv.reader(r, delimiter=',', quotechar='|')
        count = sum(1 for row in inp)
        print (count)
        print (count % 17)


a = 'Compile/imputed.csv'
b = 'incomplete.csv'
c = 'Compile/Complete_Tal.csv'

#concac(a, b)
countRow('compile/predicted.csv')
countRow('sorted_predicted.csv')