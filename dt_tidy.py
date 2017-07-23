''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_tidy.py

Prepare input data for DNN
1. Remove CusId
2. merge FetchDate and 1stContract by introducing CusTime = (FetchDate - 1stContract )
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
''' 

import csv

def  tidy_dt(data):
	with open(data, 'r', newline='') as rf, open("data2.csv", 'w', newline='') as wf:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None)	# read headers
		fields[0] = "CusTime"
		del fields[6]	# remove 1stContract
		del fields[1]	# remove CusId
		writer.writerow(fields)	# write headers

		for row in reader: 
			row[0] = int(row[0]) - int(row[6])	# CusTime = (FetchDate - 1stContract )
			del row[6]	# remove 1stContract
			del row[1]	# remove CusId
			writer.writerow(row)

tidy_dt("features2.csv")