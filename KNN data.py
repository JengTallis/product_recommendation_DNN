import csv

def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1

def Min (m, exist):
    minim = []
    for x in exist:
        minim.append(abs(int(m)-int(x)))
    minimum = min(minim)
    point = minim.index(minimum)
    date = exist[point]
    return date

def missing (test):
    miss = ['16949', '16919', '16888', '16859', '16828', '16797', '16767', '16736', '16706', '16675', '16644', '16614', '16583', '16553', '16522', '16494', '16463']
    exist = []
    out = []
    for t in test:
        exist.append(t[0])
        if checkList(miss, t[0]) != -1:
            miss.remove(t[0])
    for m in miss:
        o = []
        date = Min(m, exist)
        for t in test:
            if t[0] == date:
                for i in range (18):
                    o.append(t[i])
                o[0] = m
        out.append(o)
    return out

def data (file):
    with open(file, 'r') as r, open("missing.csv", 'w', newline='') as wr:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')
        fn = next(inp)
        out.writerow(fn)
        test = []
        for row in inp:
            if len(test) == 0:
                test.append(row)
            else:
                if row[1] == test[len(test) - 1][1]:
                    test.append(row)
                else:
                    miss = missing(test)
                    for x in miss:
                        out.writerow(x)
                    test=[]
                    test.append(row)

data('incomplete.csv')