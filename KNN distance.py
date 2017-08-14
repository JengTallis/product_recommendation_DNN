import pandas as pd
import csv

def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1
    else:
        return list.index(input)

def countCust(file):
    with open(file, 'r') as r:
        inp = csv.reader(r, delimiter=',', quotechar='|')
        count = 0
        count = sum(1 for row in inp)

        print (count)
        print (count % 17)

def appendId(fun, test, row, fn, out):
    if len(test) == 0:
        test.append(row)
    else:
        if row[1] == test[len(test) - 1][1]:
            test.append(row)
        else:
            out = fun(test, fn)
            test = []
            test.append(row)
            return test, out
    return test, out

def aveSegment(test, fn):
    ave = 0
    idx = fn.index('Segment')
    for row in test:
        ave += int(float(row[idx]))
    ave = round(ave/17, 0)
    return int(ave)


def divideSeg(file, seg):
    with open(file , 'r') as r, open (seg[0], 'w', newline = '') as w1, open (seg[1], 'w', newline = '') as w2, open (seg[0], 'w', newline = '') as w3:

        inp = csv.reader(r, delimiter=',', quotechar='|')
        f1 = csv.writer(w1, delimiter=",", quotechar='|')
        f2 = csv.writer(w2, delimiter=",", quotechar='|')
        f3 = csv.writer(w3, delimiter=",", quotechar='|')
        f = [f1,f2,f3]

        fn = next(inp)
        f1.writerow(fn)
        f2.writerow(fn)
        f3.writerow(fn)

        test = []
        out = 0

        for row in inp:
            if len(test) == 0:
                test.append(row)
            else:
                if row[1] == test[len(test) - 1][1]:
                    test.append(row)
                else:
                    ave = 0
                    idx = fn.index('Segment')
                    for t in test:
                        ave += int(float(t[idx]))
                    ave = round(ave / 17, 0) - 1
                    for t in test:
                        f[int(ave)].writerow(t)
                    test = []
                    test.append(row)

def main (comp, impu, i):
    with open(comp[i] , 'r') as r1, open (impu[i], 'r') as r2:

        comp = csv.reader(r1, delimiter=',', quotechar='|')
        impu = csv.reader(r2, delimiter=",", quotechar='|')

        fn = next(inp)


        test = []

a = 'sorted predicted.csv'
b = 'complete.csv'

segFileComp = ['comp1.csv', 'comp2.csv', 'comp3.csv']
segFileImpu = ['impu1.csv', 'impu2.csv', 'imou3.csv']
divideSeg(a, segFileImpu)
divideSeg(b, segFileComp)
countCust(segFileImpu[0])
countCust(segFileImpu[1])
countCust(segFileComp[0])
countCust(segFileComp[1])