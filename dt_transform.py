'''
dt_transform.py

Transform the clean data set to numbers

'''

import csv
import datetime

def transform_dt(data):

    # open read_file and write_file
    with open(data, 'r', newline='') as rf, open("num.csv", 'w', newline='') as wf:

        reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
        writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

        fields = next(reader, None) # read headers
        writer.writerow(fields) # write headers

        # transform fields to numeric values
        for row in reader:
            
            # FetchDate
            date = datetime.datetime.strptime(row[fields.index('FetchDate')], '%Y-%m-%d')
            row[fields.index('FetchDate')] = int(date.timestamp()/86400) # POSIX timestamp, seconds from 1970-1-1

            # EmployeeIdx
            EmployeeIdx = {'A' : 0,     # active
                           'S' : 0,
                           'B' : 1,     # ex-employed
                           'F' : 2,     # filial
                           'N' : 3,     # not Employee
                           'P' : 4}     # pasive
            row[fields.index('EmployeeIdx')] = EmployeeIdx[row[fields.index('EmployeeIdx')]]

            # CntyOfResidence
            CntyOfResidence = []
            cnty = row[fields.index('CntyOfResidence')]
            if cnty in CntyOfResidence:
                row[fields.index('CntyOfResidence')] = CntyOfResidence.index(cnty)
            else:
                CntyOfResidence.append(cnty)
                row[fields.index('CntyOfResidence')] = CntyOfResidence.index(cnty)

            # Sex
            Sex = {'H' : 0,             #Male
                   'V' : 1}             #Female
            row[fields.index('Sex')] = Sex[row[fields.index('Sex')]]

            # 1stContract
            date = datetime.datetime.strptime(row[fields.index('1stContract')], '%Y-%m-%d')
            row[fields.index('1stContract')] = int(date.timestamp()/86400)

            # RelationType
            RelationType = {'A' : 0,    #Active
                            'I' : 1,    #Inactive
                            'P' : 2,    #Former Customer
                            'R' : 3}    #Potential
            row[fields.index('RelationType')] = RelationType[row[fields.index('RelationType')]]

            # ForeignIdx
            ForeignIdx = {'N' : 0,
                          'S' : 1}
            row[fields.index('ForeignIdx')] = ForeignIdx[row[fields.index('ForeignIdx')]]

            # ChnlEnter
            ChnlEnter= []
            chnl = row[fields.index('ChnlEnter')]
            if chnl in ChnlEnter:
                row[fields.index('ChnlEnter')]
            else:
                ChnlEnter.append(chnl)
                row[fields.index('ChnlEnter')] = ChnlEnter.index(chnl)

            # DeceasedIdx
            DeceasedIdx = {'N' : 0,
                           'S' : 1}
            row[fields.index('DeceasedIdx')] = DeceasedIdx[row[fields.index('DeceasedIdx')]]

            # Income
            row[fields.index('Income')] = round(float(row[fields.index('Income')]),2)

            # Segment
            seg = row[fields.index('Segment')].split(' ')    #  1 - VIP, 2 - Individuals 3 - college graduated
            row[fields.index('Segment')] = int(seg[0])

            # Nonenegative
            none_neg = True
            for i in range (len(row)):
                if type(row[i]) == str :
                    row[i] = int(float(row[i]))
                if row[i] < 0:
                    none_neg = False

            if none_neg:
                writer.writerow(row)

transform_dt("cln.csv")