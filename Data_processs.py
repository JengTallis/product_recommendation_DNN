import tensorflow as tf
import pandas as pd
import csv
import datetime as dt
import numpy

def ProcessData(Input):
    with open(Input, 'r') as f, open ("cleansed.csv", 'w', newline= '') as wr:
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
                out.writerow(row)

ProcessData("train_ver2.csv")