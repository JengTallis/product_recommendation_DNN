'''''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_sort.py

Sort the data by cusID and then FetchDate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

		'''
		sort = sorted(reader, key=operator.itemgetter(1,0)) # sort as string
		'''

		sort = sorted(reader, key= lambda row: int(row[1])) # stable sort as number
		#sort = sorted(sort, key= lambda row: int(row[0])) 

		for row in sort:
			writer.writerow(row)

sort_dt("num.csv")