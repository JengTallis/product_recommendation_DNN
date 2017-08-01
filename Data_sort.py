import pandas as pd
import csv
import collections


def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1

def months (file):
    m = []
    with open(file, 'r') as r:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        field = next(inp)
        for row in inp:
            if checkList(m, row[0]) == -1:
                m.append(row[0])
    sorted(m)
    print (m)

def custMonthly(file):
    print (str(file))
    m = ['16463', '16494', '16522', '16553', '16583', '16614', '16644', '16675', '16706', '16736', '16767', '16797', '16828', '16859', '16888', '16919', '16949']
    count = []
    for i in range(len(m)):
        count.append(0)
    i=1
    with open(file, 'r') as r:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        field = next(inp)
        total = 0
        for row in inp:
            count[m.index((row[0]))] +=1
        for x in range(len(count)):
            if x>0:
                print(str(i) + '. ' + str(m[x]) + ': ' + str(count[x]) + ', diff: ' + str(count[x] - count[x-1]))
            else:
                print(str(i) + '. ' + str(m[x]) + ': ' + str(count[x]))

            i+=1
        for i in count:
            total += i
        #print ('total number of records =' + str(total))
        print( '\n')


def  sortData (filein):
    df = pd.read_csv(filein)
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, False])
    df.to_csv('sorted.csv',index=False, sep = ',', encoding = 'utf-8')

def completeData(file):
    with open(file, 'r') as r, open("complete.csv", 'w', newline='') as wr1, open("incomplete.csv", 'w', newline='') as wr2:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out1 = csv.writer(wr1, delimiter=",", quotechar='|')
        out2 = csv.writer(wr2, delimiter=",", quotechar='|')
        fn = next(inp)
        print (fn)
        out1.writerow(fn)
        out2.writerow(fn)
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
                            out1.writerow(t)
                    else:
                        for t in test:
                            out2.writerow(t)
                    test = []
                    test.append(row)

def consecutive (test):
    m = ['16949', '16919', '16888', '16859', '16828', '16797', '16767', '16736', '16706', '16675', '16644', '16614', '16583', '16553', '16522', '16494', '16463']
    con = True
    num = len(test)
    init = test[0][0]
    start = m.index(init)
    last = start + num
    for i in range (start, last):
        if m[i] != test[i-start][0]:
            con = False
    return con

def incompleteData (file):
    with open(file, 'r') as r, open("consecutive.csv", 'w', newline='') as wr1, open("inconsecutive.csv", 'w',newline='') as wr2:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out1 = csv.writer(wr1, delimiter=",", quotechar='|')
        out2 = csv.writer(wr2, delimiter=",", quotechar='|')
        fn = next(inp)
        print(fn)
        out1.writerow(fn)
        out2.writerow(fn)
        test = []
        for row in inp:
            if len(test) == 0:
                test.append(row)
            else:
                if row[1]== test[len(test)-1][1]:
                    test.append(row)
                else:
                    if consecutive(test) == True:
                        for t in test:
                            out1.writerow(t)
                    else:
                        for t in test:
                            out2.writerow(t)
                    test = []
                    test.append(row)





#sortData("transformed.csv")
#completeData('sorted.csv')
#incompleteData('incomplete.csv')

custMonthly('transformed.csv')
custMonthly('sorted.csv')
custMonthly('complete.csv')
custMonthly('incomplete.csv')
custMonthly('consecutive.csv')
custMonthly('inconsecutive.csv')

