''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
csv_combine.py

Read 2 csv files from command line argument 
append the second file after the first file and ourput a new csv file

Usage:
python3 csv_combine.py csv_file_name1.csv csv_file_name2.csv

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import sys
import csv

'''
read csv file
print to command line output
'''
def combine(data, data2):
	with open(data, 'r', newline='') as rf, open(data2, 'r', newline='') as rf2, open("knn_complete.csv", 'w', newline='') as wf:
		record_cnt = 0
		field_cnt = 0

		reader1 = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader
		reader2 = csv.reader(rf2, delimiter=",", quotechar='|') # csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer

		fields = next(reader1, None)  # read skip header row, ignore return headers
		next(reader2, None) 

		writer.writerow(fields) # write header row

		writer.writerows(reader1)
		writer.writerows(reader2)

# read file from command line argument
# argv[0] is this file name (csv_reader.py)
file1 = sys.argv[1]
file2 = sys.argv[2]
combine(file1, file2)