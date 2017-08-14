from sklearn.neighbors import NearestNeighbors as nn
from sklearn import preprocessing as pp
import random as rd
import numpy as np
import pandas as pd
import csv

def write (data, testFile, fn):
    with open (testFile, 'w', newline= '') as wr:
        out = csv.writer(wr, delimiter=",", quotechar='|')
        out.writerow(fn)
        for d in data:
            out.writerow(d)
        print (testFile + ' made')

def splitData(proportion, filename):
    dic = {
        'EmplooyeeIdx':int, 'CntyOfResidence':int, 'Sex':int, 'Age':int, '1stContract':int,
        'NewCusIdx':int, 'Seniority':int, 'CusType':int, 'RelationType':int, 'ForeignIdx':int,
        'ChnlEnter':int, 'DeceasedIdx':int, 'ProvCode':int, 'ActivIdx':int, 'Income':float,
        'Segment':int,

        'SavingAcnt':int, 'Guarantees':int, 'CurrentAcnt':int, 'DerivativesAcnt':int, 'PayrollAcnt':int,
        'JuniorAcnt':int, 'MoreParticularAcnt':int, 'ParticularAcnt':int, 'ParticularPlusAcnt':int, 'ShortDeposit':int,
        'MediumDeposit':int, 'LongDeposit':int, 'eAcnt':int, 'Funds':int, 'Mortgage':int,
        'Pensions':int, 'Loans':int, 'Taxes':int, 'CreditCard':int, 'Securities':int,
        'HomeAcnt':int, 'Payroll':int, 'PayrollPensions':int, 'DirectDebit':int
           }
    fn = ["FetchDate", "CusID", "EmployeeIdx", "CntyOfResidence", "Sex",    #Fieldnames
                 "Age", "1stContract", "NewCusIdx", "Seniority", "CusType",
                 "RelationType", "ForeignIdx", "ChanEnter", "DeceasedIdx", "ProvCode",
                 "ActivIdx", "Income", "Segment",

              "SavingAcnt", "Guarantees",
                 "CurrentAcnt", "DerivativesAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                 "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                 "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                 "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                 "PayrollPensions", "DirectDebit" ]
    df = pd.read_csv(filename, header=0, dtype=dic)
    data = df.as_matrix()
    rowsNum = len(data)
    custNum = rowsNum/17
    print (rowsNum)
    for i in range (len(proportion)):
        proportion[i] = round(proportion[i]*custNum,0)*17
    output1 = data[:int(proportion[0])]
    write(output1, 'fitData.csv', fn)
    output2 = data[int(proportion[0]):]
    print('data has been split')
    return output2

def randSelect():
    out = []
    rand = rd.randint(1,15)
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    for r in range(rand):
        i = rd.randint(0, len(idx)-1)
        out.append(idx.pop(i))
    return out

def selectData(testData, testFile):
    x = testData[0][1]
    test = []
    fn = ["FetchDate", "CusID", "EmployeeIdx", "CntyOfResidence", "Sex",    #Fieldnames
                 "Age", "1stContract", "NewCusIdx", "Seniority", "CusType",
                 "RelationType", "ForeignIdx", "ChanEnter", "DeceasedIdx", "ProvCode",
                 "ActivIdx", "Income", "Segment",

              "SavingAcnt", "Guarantees",
                 "CurrentAcnt", "DerivativesAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                 "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                 "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                 "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                 "PayrollPensions", "DirectDebit" ]
    with open (testFile, 'w', newline= '') as wr:
        out = csv.writer(wr, delimiter=",", quotechar='|')
        out.writerow(fn)
        for row in testData:
            if row[1] == x:
                test.append(row)
            else:
                idx = randSelect()
                for i in idx:
                    out.writerow(test[i])
                test = []
                test.append(row)
                x = row[1]
    print('data is randomly selected')

def  sortData (filein):
    df = pd.read_csv(filein)
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, True])
    df.to_csv('complete1.csv',index=False, sep = ',', encoding = 'utf-8')

def checkList (list, input):
    try:
        list.index(input)
    except ValueError:
        return -1
    else:
        return list.index(input)

def getData(file):
    m = ['16463', '16494', '16522', '16553', '16583', '16614', '16644', '16675', '16706', '16736', '16767', '16797',
         '16828', '16859', '16888', '16919', '16949']
    dic = {
        'EmplooyeeIdx':int, 'CntyOfResidence':int, 'Sex':int, 'Age':int, '1stContract':int,
        'NewCusIdx':int, 'Seniority':int, 'CusType':int, 'RelationType':int, 'ForeignIdx':int,
        'ChnlEnter':int, 'DeceasedIdx':int, 'ProvCode':int, 'ActivIdx':int, 'Income':float,
        'Segment':int,

        'SavingAcnt':int, 'Guarantees':int, 'CurrentAcnt':int, 'DerivativesAcnt':int, 'PayrollAcnt':int,
        'JuniorAcnt':int, 'MoreParticularAcnt':int, 'ParticularAcnt':int, 'ParticularPlusAcnt':int, 'ShortDeposit':int,
        'MediumDeposit':int, 'LongDeposit':int, 'eAcnt':int, 'Funds':int, 'Mortgage':int,
        'Pensions':int, 'Loans':int, 'Taxes':int, 'CreditCard':int, 'Securities':int,
        'HomeAcnt':int, 'Payroll':int, 'PayrollPensions':int, 'DirectDebit':int
           }
    df = pd.read_csv(file, header=0, dtype=dic)
    date = df.pop('FetchDate')
    date = date.as_matrix()
    id  = df.pop('CusID')
    id = id.as_matrix()
    data2d = df.as_matrix()
    output2d = np.array(data2d)
    return id, date, output2d

def lookID(indices, comp_id):
    id = []
    out = {}
    s = 0
    for i in indices:
        for x in i:
            if checkList(id,comp_id[x]) == -1:
                id.append(comp_id[x])
                out[comp_id[x]] = 1
            else:
                out[comp_id[x]] +=1
            s +=1
    for k, v in out.items():
        out[k]=float(v/s)
    return out

def checkDraw (id_dic):
    ls = []
    for k, v in id_dic.items():
        ls.append(v)
    unique = set(ls)
    if len(unique) != len(ls):
        print ('Draw')
        return True
    else:
        print ('no draw')
        return False

def missingM(date):
    month = ['16463', '16494', '16522', '16553', '16583', '16614', '16644', '16675', '16706', '16736', '16767', '16797',
         '16828', '16859', '16888', '16919', '16949']
    out = []
    for m in month:
        if checkList(date, m) == -1:
            out.append(m)
    return out

def predictFrom(id_dic, comp_id, comp_data2d): #returns index of comp data to predict from
    out= {}
    idx = []
    for k, v in id_dic.items():
        row, = np.where(comp_id==k)
        for r in row:
            idx.append(comp_data2d[r])
        out[k]= idx
        idx = []
    return out

def calcRow(row, v, fn, test): #calculate each row of missing data
    row[(fn.index("EmployeeIdx")-2)] = round((row[(fn.index("EmployeeIdx")-2)]*v),0)
    row[(fn.index("CntyOfResidence")-2)] = round((row[(fn.index("CntyOfResidence")-2)]*v),0)
    row[(fn.index("Sex")-2)] = test[0][(fn.index('Sex')-2)]
    row[(fn.index("Age")-2)] = test[0][(fn.index("Age")-2)]
    row[(fn.index("1stContract")-2)] = row[(fn.index("1stContract")-2)]
    row[(fn.index("NewCusIdx")-2)] = round((row[(fn.index("NewCusIdx")-2)]*v),0)
    row[(fn.index("Seniority")-2)] = round((row[(fn.index("Seniority")-2)]*v),2)
    row[(fn.index("CusType")-2)] = round((row[(fn.index("CusType")-2)]*v),0)
    row[(fn.index("RelationType")-2)] = round((row[(fn.index("RelationType")-2)]*v),0)
    row[(fn.index("ForeignIdx")-2)] = round((row[(fn.index("ForeignIdx")-2)]*v),0)
    row[(fn.index("ChanEnter")-2)] = row[(fn.index("ChanEnter")-2)]
    row[(fn.index("DeceasedIdx")-2)] = round((row[(fn.index("DeceasedIdx")-2)]*v),0)
    row[(fn.index("ProvCode")-2)] = round((row[(fn.index("ProvCode")-2)]*v),0)
    row[(fn.index("ActivIdx")-2)] = round((row[(fn.index("ActivIdx")-2)]*v),0)
    row[(fn.index("Income")-2)] = round((row[(fn.index("Income")-2)]*v),2)
    row[(fn.index("Segment")-2)] = round((row[(fn.index("Segment")-2)]*v),0)
    return row

def closeDate(month, missing_m, miss):
    x = 1000
    out = 0
    idx = checkList(month, miss)
    for i in range (idx):
        if checkList(missing_m, int(month[i])) == -1:
            if x > (miss - int(month[i])):
                x = (miss-int(month[i]))
                out = int(month[i])
    return out

def predMiss(cusID, missing_m, id_dic, predData, fn, test):
    month = ['16463', '16494', '16522', '16553', '16583', '16614', '16644', '16675', '16706', '16736', '16767', '16797',
         '16828', '16859', '16888', '16919', '16949']
    output = []
    for miss in missing_m: # for each missing month
        test1 = [] # matrix to calculate weights
        test2 = [] # matrix to sum the weights
        prop = []

        idx = month.index(str(miss)) # Find index for stated missing month
        for k, v in id_dic.items(): # For each constituent of the predicted data
            test1.append(predData[k][idx])
            prop.append(v)
        for i in range(len(test1)):
            test2.append(calcRow(test1[i], prop[i], fn, test))
        out = [0] * 40
        for r in range(len(test2)):
            for c in range(len(fn)-2):
                out[c] += test2[r][c]
        out.insert(0, cusID)
        out.insert(0, miss)
        output.append(out)
    return output

def main():

    proportion = [0.9, 0.1]
    oriFile = 'complete1.csv'
    trainFile = 'fitData.csv'
    testFile = 'test1.csv'

    '''
    trainfile = 'complete.csv'
    testFile = 'incomplete.csv'
    '''

    selectData(splitData(proportion, oriFile), testFile)   # splitting to test for accuracy
    sortData(testFile)

    comp_id, comp_date, comp_data2d = getData(trainFile)
    nbrs = nn(n_neighbors=2, algorithm='auto').fit(comp_data2d)

    test = []
    date = []
    id = []
    x = 0
    with open(testFile, 'r') as f, open ("pred1.csv", 'w', newline= '') as wr:
        inp = csv.reader(f, skipinitialspace=True, delimiter=',', quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')
        fn = next(inp)
        out.writerow(fn)
        for row in inp:
            date.append(row.pop(0))
            id.append(row.pop(0))
            if x == 0:
                x = id[len(id)-1]
            if x == id[len(id)-1]:
                test.append(row)
            else:
                distances, indices = nbrs.kneighbors(test)
                id_dic = lookID(indices, comp_id)
                predData = predictFrom(id_dic,comp_id,comp_data2d)
                CusID = id[0]


                missing_m = missingM(date)
                o = predMiss(CusID, missing_m, id_dic, predData, fn, test)
                for ans in o:
                    out.writerow(ans)

                test = []
                test.append(row)
                date = [date[len(date)-1]]
                id = [id[len(id)-1]]
                x = id[len(id)-1]
                break

def accuracy():
    

main()