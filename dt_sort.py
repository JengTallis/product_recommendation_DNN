'''
dt_sort.py

Sort the data by cusID and then FetchDate

'''
import operator
import csv

def  sort_dt(data):
    with open(data, 'r', newline='') as rf, open("sort.csv", 'w', newline='') as wf:

        reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
        writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

        fields = next(reader, None) # read headers
        writer.writerow(fields) # write headers

        # sort the data by cusID(1) and then FetchDate(0)
        sort = sorted(reader, key=operator.itemgetter(1,0))

        for row in sort:
            writer.writerow(row)

sort_dt("num.csv")

