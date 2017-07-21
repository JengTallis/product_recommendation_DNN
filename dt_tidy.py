''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_tidy.py

Prepare input data for DNN
0. Remove header
1. Remove cusId
2. merge 1stContract and FetchDate by introducing cusTime = (1stContract - FetchDate)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
''' 

import csv

def  tidy_dt(data):
	with open(data, 'r', newline='') as rf, open("sort.csv", 'w', newline='') as wf:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None) # read headers
		writer.writerow(fields) # write headers

		for row in reader:
			writer.writerow(row)

sort_dt("num.csv")