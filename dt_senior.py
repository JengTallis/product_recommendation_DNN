'''''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_senior.py

Remove customer who doesn't have data for every month (join before the first month)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import csv

def dt_senior(data):

	with open(data, 'r', newline='') as rf, open("senior.csv", 'w', newline='') as wf, open("incomplete.csv", 'w', newline='') as wf2:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer
		writer2 = csv.writer(wf2, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None) # read headers
		writer.writerow(fields) # write headers
		writer2.writerow(fields)

		# remove customers with less than n_months records
		
		# cnt = 0
		n_months = 17
		cus = 0
		cus_dt = []
		retain = 0
		disgard = 0

		for row in reader:
			if len(cus_dt) > 0 and int(row[1]) != cus :
				if len(cus_dt) == n_months: # complete customer
					#print("Customer %d has complete 17 months" %cus)
					for r in cus_dt:
						writer.writerow(r)
					retain += 1
				else:	
					for r in cus_dt:	# incomplete customer
						writer2.writerow(r)	
					disgard += 1
				cus_dt = []	# clear buffer			
			cus_dt.append(row)
			cus = int(row[1])
		print("Total number of cusomers: %d" %(retain+disgard))
		print("Complete customers #: %d" %retain)
		print("Incomplete customers #: %d" %disgard)
		print("Complete ratio: %f" %float(retain/(retain+disgard)))

def make_missing_months(data):
	months=[16462,16493,16521,16552,16582,16613,16643,16674,16705,16735,16766,16796,16827,16858,16887,16918,16948]

	with open(data, 'r', newline='') as rf, open("missing.csv", 'w', newline='') as wf:

		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader, None) # read headers
		writer.writerow(fields) # write headers

		cust = []
		c_info = []
#		m = 0

		for row in reader:
			if not (len(cust) == 0 or row[1] == cust[len(cust)-1][1]): # cust is a customer batch

				r = cust.pop(0)
				c_info = r[0:18]
				for m in range(len(months)):
					if m == months.index(int(r[0])):
						if len(cust) > 0 :
							r = cust.pop(0)
							c_info = r[0:18]
					else:
						c_info[0] = months[m]
						writer.writerow(c_info)
			cust.append(row)

#dt_senior("sort.csv")
make_missing_months("incomplete.csv")