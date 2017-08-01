'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_clean.py

Clean up the given data set:

Remove unnecessary fields (cols)
Remove incomplete records (rows)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import csv

def clean_dt(data):

    # Define fields for a record
    fields = ["FetchDate", "CusID", "EmployeeIdx", "CntyOfResidence", "Sex",    
                "Age", "1stContract", "NewCusIdx", "Seniority", "CusType",
                "RelationType", "ForeignIdx", "ChnlEnter", "DeceasedIdx", "ProvCode",
                "ActivIdx", "Income", "Segment", "SavingAcnt", "Guarantees",
                "CurrentAcnt", "DerivativeAcnt", "PayrollAcnt", "JuniorAcnt", "MoreParticularAcnt",
                "ParticularAcnt", "ParticularPlusAcnt", "ShortDeposit", "MediumDeposit", "LongDeposit",
                "eAcnt", "Funds", "Mortgage", "Pensions", "Loans",
                "Taxes", "CreditCard", "Securities", "HomeAcnt", "Payroll",
                "PayrollPensions", "DirectDebit"]

    # open read_file and write_file
    with open(data, 'r', newline='') as rf, open ("cln.csv", 'w', newline='') as wf: 
        
        reader = csv.reader(rf, skipinitialspace=True, delimiter=',', quotechar='|') # csv_reader
        writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

        next(reader, None)  # read skip header row, ignore return headers
        writer.writerow(fields) # write header row
        writer2.writerow(fields) # write header row

        for row in reader:

            # remove useless fields (cols)
            del row[9]  # indrel
            del row[9]  # ult_fec_cl_1t
            del row[11] # indresi
            del row[12] # conyuemp
            del row[14] # tipodom
            del row[15] # nomprov
            if (len(row) > 42): # longer than 24 + 24 - 6 = 42 fields, misparsed records
                del row[15] # second half of nomprov with a "," 

            # check for incomplete records
            incomplete = False
            for fld in row:
                if (fld == '') or (fld == "NA"):
                    incomplete = True
                    break

            # retain complete records
            if not incomplete: 
                writer.writerow(row)

clean_dt("train.csv")