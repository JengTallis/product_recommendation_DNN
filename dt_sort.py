'''
dt_sort.py

Sort the data by cusID and then FetchDate
Remove customer who doesn't have data for every month (join before the first month)

'''

import operator
import csv

def  sort_dt(data):
	with open(data, 'r', newline='') as rf, open("sort.csv", 'w', newline='') as wf:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None) # read headers
		writer.writerow(fields) # write headers

		# sort the data by cusID(1) and then FetchDate(0) in ascending order
		sort = sorted(reader, key=operator.itemgetter(1,0))

		# remove customers with less than n_months records
		n_months = 17
		cnt = 0
		cus = 0
		customers = []

		for row in sort:
			if row[1] != cus: # an unseen customer
				cus_dt = []
				cus = row[1]
				cnt = 0
			cnt += 1
			cus_dt.append(row)
			if (cnt >= n_months): # wanted customer
				for r in cus_dt:
					writer.writerow(r)

sort_dt("num.csv")