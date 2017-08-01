import csv
import datetime as dt

def work():
    print('works')

def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1

def Month():
    i = ''
    mList = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
    print('Which month to predict?')
    print(mList)
    i = input('Answer:')
    while checkList(mList,i)==-1:
        print('Try again.')
        i = input('Answer:')
    i = mList.index(i)+1
    return i

def checkLast(row, i):
    x = dt.datetime.utcfromtimestamp(float(row[0])*86400)
    y = dt.datetime.strftime(x, '%m')
    z = dt.datetime.strftime(x, '%y')
    if int(y) * int(z) == ((i-1)*16):
        return True

def checkPred(row, i):      # to check if month is predicted month
    x = dt.datetime.utcfromtimestamp(float(row[0])*86400)
    y = dt.datetime.strftime(x, '%m')
    z = dt.datetime.strftime(x, '%y')
    if int(y) * int(z) == (i*16):
        return True
    else:
        return False

def sameID (row, test):
    if len(test) == 0:
        test.append(row)
    else:
        if row[1] == test[len(test) - 1][1]:
            test.append(row)
    return (test)


def PFchange(test, field):      #2 and #4
    state = [[0,0],[0,1],[1,0],[1,1]]
    idxChange = []

    for c in range(field.index('SavingAcnt'), (field.index('DirectDebit') + 1)):
        s1, s2, s3, s4  = 0, 0, 0, 0
        S = {0:s1, 1:s2, 2:s3, 3:s4}
        x1, xm, y0, ym= 1, 0, 1, 0

        for r in range(len(test)-1):
            m1 = int(test[r][c])
            m2 = int(test[r+1][c])
            m3 = [m1,m2]

            S[state.index(m3)] +=1

            if m1 - m2 == 0:
                if m1 == 1:
                    x1 +=1
                    if x1 > xm:
                        xm = x1
                else:
                    y0 +=1
                    if y0 > ym:
                        ym = y0
            else:
                x1 = 1
                y0 = 1

        count = []
        for k, v in S.items():
            count.append(v)
        count.append(ym)
        count.append(xm)
        idxChange.append(count)
    return idxChange


def fieldname(file):
    with open(file, 'r') as f:
        inp=csv.reader(f, delimiter= ",", quotechar='|')
        field = next(inp)
        pf = ['00', '01', '10', '11', 'M0', 'M1']

        for i in range (field.index('SavingAcnt'), (field.index('DirectDebit') + 1)):
            for p in pf:
                field.append(p + '_' + field[i])

        for i in range(field.index('SavingAcnt'), (field.index('DirectDebit') + 1)):
            field.append('PR_' + field[i])


        field.insert(0,field[1])
        del field[2]
        del field[6]
        field[1] = 'DaysWithBank'
        for i in range(field.index('DirectDebit')+1):
            field[i] = "LM_" + field[i]
        return field

def PFonly (row, field):
    for i in range (field.index('SavingAcnt')):
        del row[0]
    return row

def FeaturizeData (file):
    i = Month()

with open(file, 'r') as r, open("featurized.csv", 'w', newline='') as wr:
    inp = csv.reader(r, delimiter=",", quotechar='|')
    out = csv.writer(wr, delimiter=",", quotechar='|')

    field = next(inp)
    header = fieldname(file)
    out.writerow(header)

    label = []
    test = []
    lst = []
    pf = []

    for row in inp:
        outrow = []
        if checkLast(row, i) == True:
            for r in row:
                if float(r) % 1 > 0:
                    lst.append(round(float(r),2))
                else:
                    lst.append(int(float(r)))

        if checkPred(row, i) == False:
            test = sameID(row, test)
        else:
            row = PFonly(row, field)
            for r in row:
                label.append(r)

        if len(test) == 16:
            P = []
            P = PFchange(test, field)
            for p1 in P:
                for p2 in p1:
                    pf.append(p2)
            outrow = lst + pf + label
            pf = []
            outrow.insert(0, outrow[1])
            del outrow[2]
            outrow[1] = outrow[1] - outrow[6]
            del outrow[6]
            out.writerow(outrow)
            label = []
            test = []
            lst = []
FeaturizeData("complete.csv")
