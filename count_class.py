# count_class.py

import csv

def count_class(data):

    # Define products
    product_names = ["SavingAcnt        ", "Guarantees         ","CurrentAcnt         ", "DerivativeAcnt      ",
                    "PayrollAcnt       ", "JuniorAcnt        ", "MoreParticularAcnt", "ParticularAcnt      ", 
                    "ParticularPlusAcnt", "ShortDeposit      ", "MediumDeposit     ", "LongDeposit        ",
                    "eAcnt             ", "Funds             ", "Mortgage          ", "Pensions           ", 
                    "Loans             ", "Taxes             ", "CreditCard        ", "Securities         ", 
                    "HomeAcnt          ", "Payroll           ", "PayrollPensions   ", "DirectDebit        "]

    product_start_idx = 18

    # open read_file and write_file
    with open(data, 'r', newline='') as rf: 
        
        reader = csv.reader(rf, skipinitialspace=True, delimiter=',', quotechar='|') # csv_reader
        next(reader, None)  # read skip header row, ignore return headers

        products = [[0, 0] for j in range(len(product_names))]  # initialize class counts

        # count class occurrences for each product
        for row in reader:
            for i in range(product_start_idx, product_start_idx+len(product_names)):
                products[i-product_start_idx][int(row[i])] += 1

        #print(products)

        for i in range(len(products)):
            print("Product %d %s " %(i, product_names[i]), end='\t')
            print("Class:   ", products[i], end='\t')
            print("Ratio neg/pos:   %f" %(float(products[i][0])/float(products[i][1])), end='\t')
            print("Sum %d" %(int(products[i][0])+int(products[i][1])))
        

count_class("senior.csv")