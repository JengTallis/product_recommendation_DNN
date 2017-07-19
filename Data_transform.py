import csv
import datetime as dt

def  processData (file):
    with open(file, 'r') as r, open("transformed.csv", 'w', newline='') as wr:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')
        fn = next(inp)
        out.writerow(fn)
        for row in inp:
            i = True

            #FetchDate
            format = '%Y-%m-%d'
            strDate = row[fn.index('FetchDate')]
            date = dt.datetime.strptime(strDate, format)
            epoch = dt.datetime.utcfromtimestamp(0)
            row[fn.index('FetchDate')] = int(((date - epoch).total_seconds())/86400)

            #EmployeeIdx
            EmployeeIdx = {'A' : 0,     #Active
                           'S' : 0,
                           'B' : 1,     #ex-employed
                           'F' : 2,     #Filial
                           'N' : 3,     #Not Employee
                           'P' : 4}     #Passive
            row[fn.index('EmployeeIdx')] = EmployeeIdx[row[fn.index('EmployeeIdx')]]

            #CntyOfResidence
            CntyOfResidence = []
            idx = -1
            for cnty in CntyOfResidence:
                if row[fn.index('CntyOfResidence')] == cnty:
                    idx = CntyOfResidence.index(cnty)
            if idx!=-1:
                row[fn.index('CntyOfResidence')] = idx
            else:
                CntyOfResidence.append(row[fn.index('CntyOfResidence')])
                row[fn.index('CntyOfResidence')] = CntyOfResidence.index(row[fn.index('CntyOfResidence')])

            #Sex
            Sex = {'H' : 0,             #Male
                   'V' : 1}             #Female
            row[fn.index('Sex')] = Sex[row[fn.index('Sex')]]

            #1stContract
            strDate = row[fn.index('1stContract')]
            date = dt.datetime.strptime(strDate, format)
            row[fn.index('1stContract')] = int(((date - epoch).total_seconds())/86400)

            #RelationType
            RelationType = {'A' : 0,    #Active
                            'I' : 1,    #Inactive
                            'P' : 2,    #Former Customer
                            'R' : 3}    #Potential
            row[fn.index('RelationType')] = RelationType[row[fn.index('RelationType')]]

            #ForeignIdx
            ForeignIdx = {'N' : 0,
                          'S' : 1}
            row[fn.index('ForeignIdx')] = ForeignIdx[row[fn.index('ForeignIdx')]]

            #ChanEnter
            ChanEnter= []
            idx = -1
            for chan in ChanEnter:
                if row[fn.index('ChanEnter')] == chan:
                    idx = ChanEnter.index(chan)
            if idx!=-1:
                row[fn.index('ChanEnter')] = idx
            else:
                ChanEnter.append(row[fn.index('ChanEnter')])
                row[fn.index('ChanEnter')] = ChanEnter.index(row[fn.index('ChanEnter')])

            #DeceasedIdx
            DeceasedIdx = {'N' : 0,
                           'S' : 1}
            row[fn.index('DeceasedIdx')] = DeceasedIdx[row[fn.index('DeceasedIdx')]]

            #Income
            row[fn.index('Income')] = round((float(row[fn.index('Income')])),2)


            #Segment
            seg = row[fn.index('Segment')].split(' ')    #  1 - VIP, 2 - Individuals 3 - college graduated
            row[fn.index('Segment')] = int(seg[0])

            #Integers
            for l in range (len(row)-1):
                if type(row[l]) == str :
                    row[l] = int(float(row[l]))
                if row[l] < 0:
                    i = False

            if i == True:
                out.writerow(row)


processData("cleansed.csv")