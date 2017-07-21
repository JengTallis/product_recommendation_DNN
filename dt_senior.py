'''''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_senior.py

Remove customer who doesn't have data for every month (join before the first month)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import csv

def dt_senior(data):
	with open(data, 'r', newline='') as rf, open("senior.csv", 'w', newline='') as wf:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None) # read headers
		writer.writerow(fields) # write headers

		# remove customers with less than n_months records
		
		# cnt = 0
		n_months = 17
		cus = 0
		cus_dt = []

		for row in reader:
			if len(cus_dt) > 0 and int(row[1]) != cus :
				if len(cus_dt) == n_months: # wanted customer
					#print("Customer %d has complete 17 months" %cus)
					for r in cus_dt:
						writer.writerow(r)
				cus_dt = []	# clear buffer			
			cus_dt.append(row)
			cus = int(row[1])

''' BUGGY VERSION
			if row[1] != cus: # an unseen customer
				cus_dt = []
				cus = int(row[1])
				cnt = 0
			cnt += 1
			cus_dt.append(row)
			if (cnt >= n_months): # wanted customer
				for r in cus_dt:
					writer.writerow(r)
'''

dt_senior("sort.csv")