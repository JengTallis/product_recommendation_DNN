import tensorflow as tf
import pandas as pd
import csv
import datetime as dt
import numpy

def ProcessData(Input):
    with open(Input, 'r') as f, open ("TF.csv", 'w', newline= '') as wr:
        fn = ["FetchDate", "CusID", "EmployeeIdx", "CntyOfResidence", "Sex",    #Fieldnames
                 "Age", "1stContract", "NewCusIdx", "Seniority", "CusType",
                 "RelationType", "ForeignIdx", "ChanEnter", "DeceasedIdx", "ProvCode",
                 "ActivIdx", "Income", "Segment", "SavingAcnt", "Guarantees",
                 "CurrentAcnt", "DerivativesAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                 "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                 "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                 "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                 "PayrollPensions", "DirectDebit" ]
        test = csv.reader(f, skipinitialspace=True , delimiter =',', quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')
        out.writerow(fn)
        f.readline()
        x = 0

        for row in test:
            null = False

            # removing unnecessary column
            del row[9]
            del row[9]
            del row[11]
            del row[12]
            del row[14]
            del row[15]
            if (len(row)>42):
                del row[15]

            for i in row:
                if (i == '') or (i == "NA"):
                    null = True
            if null == False:
                i = True

                # FetchDate
                format = '%Y-%m-%d'
                strDate = row[fn.index('FetchDate')]
                date = dt.datetime.strptime(strDate, format)
                epoch = dt.datetime.utcfromtimestamp(0)
                row[fn.index('FetchDate')] = int(((date - epoch).total_seconds()) / 86400)

                # EmployeeIdx
                EmployeeIdx = {'A': 0,  # Active
                               'S': 0,
                               'B': 1,  # ex-employed
                               'F': 2,  # Filial
                               'N': 3,  # Not Employee
                               'P': 4}  # Passive
                row[fn.index('EmployeeIdx')] = EmployeeIdx[row[fn.index('EmployeeIdx')]]

                # CntyOfResidence
                CntyOfResidence = []
                idx = -1
                for cnty in CntyOfResidence:
                    if row[fn.index('CntyOfResidence')] == cnty:
                        idx = CntyOfResidence.index(cnty)
                if idx != -1:
                    row[fn.index('CntyOfResidence')] = idx
                else:
                    CntyOfResidence.append(row[fn.index('CntyOfResidence')])
                    row[fn.index('CntyOfResidence')] = CntyOfResidence.index(row[fn.index('CntyOfResidence')])

                # Sex
                Sex = {'H': 0,  # Male
                       'V': 1}  # Female
                row[fn.index('Sex')] = Sex[row[fn.index('Sex')]]

                # 1stContract
                strDate = row[fn.index('1stContract')]
                date = dt.datetime.strptime(strDate, format)
                row[fn.index('1stContract')] = int(((date - epoch).total_seconds()) / 86400)

                # RelationType
                RelationType = {'A': 0,  # Active
                                'I': 1,  # Inactive
                                'P': 2,  # Former Customer
                                'R': 3}  # Potential
                row[fn.index('RelationType')] = RelationType[row[fn.index('RelationType')]]

                # ForeignIdx
                ForeignIdx = {'N': 0,
                              'S': 1}
                row[fn.index('ForeignIdx')] = ForeignIdx[row[fn.index('ForeignIdx')]]

                # ChanEnter
                ChanEnter = []
                idx = -1
                for chan in ChanEnter:
                    if row[fn.index('ChanEnter')] == chan:
                        idx = ChanEnter.index(chan)
                if idx != -1:
                    row[fn.index('ChanEnter')] = idx
                else:
                    ChanEnter.append(row[fn.index('ChanEnter')])
                    row[fn.index('ChanEnter')] = ChanEnter.index(row[fn.index('ChanEnter')])

                # DeceasedIdx
                DeceasedIdx = {'N': 0,
                               'S': 1}
                row[fn.index('DeceasedIdx')] = DeceasedIdx[row[fn.index('DeceasedIdx')]]

                # Income
                row[fn.index('Income')] = round((float(row[fn.index('Income')])), 2)

                # Segment
                seg = row[fn.index('Segment')].split(' ')  # 1 - VIP, 2 - Individuals 3 - college graduated
                row[fn.index('Segment')] = int(seg[0])

                # Integers
                for l in range(len(row) - 1):
                    if type(row[l]) == str:
                        row[l] = int(float(row[l]))
                    if row[l] < 0:
                        i = False

                if i == True:
                    out.writerow(row)
    df = pd.read_csv('TF.csv')
    df = df.sort_values(['CusID','FetchDate'], ascending = [True, False])
    df.to_csv('SRT.csv',index=False, sep = ',', encoding = 'utf-8')

ProcessData("train_ver2.csv")