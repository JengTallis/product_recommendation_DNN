''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
csv_reader.py

Read csv file from command line argument 
Print to command line output

Usage:
python3 csv_reader.py csv_file_name.csv

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import sys
import csv

def read_dt(data):
	with open(data, 'r', newline='') as rf:
		record_cnt = 0
		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader

		for row in reader:
			print("Record # %d" %record_cnt)
			field_cnt = 0
			for field in row:
				print("\t Field # %d \t %s" %(field_cnt,field))
				field_cnt += 1
			record_cnt += 1

file = sys.argv[1]
read_dt(file) 