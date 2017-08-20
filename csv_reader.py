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

'''
read csv file Field # 0 	 16858
	 Field # 1 	 15889
Record # 15
	 Field # 0 	 16887
	 Field # 1 	 15889
Record # 16
	 Field # 0 	 16918
	 Field # 1 	 15889
Record # 17
	 Field # 0 	 16948
	 Field # 1 	 15889
print to command line output
'''
def read_dt(data):
	with open(data, 'r', newline='') as rf: #, open("predict.csv", 'w', newline='') as wf:
		record_cnt = 0
		field_cnt = 0
		reader = csv.reader(rf, delimiter=",", quotechar='|') # csv_reader

		#writer = csv.writer(wf, delimiter=",", quotechar='|') # csv_writer
		#fields = next(reader, None)
		#writer.writerow(fields)

		for row in reader:
			if record_cnt < 2: 
				print("Record # %d" %record_cnt)
				#writer.writerow(row)
				#print("Record # %d" %record_cnt)
			field_cnt = 0
			for field in row:
				if record_cnt < 2: print("\t Field # %d \t %s" %(field_cnt,field))
					#print("\t Field # %d \t %s" %(field_cnt,field))
				field_cnt += 1
			record_cnt += 1
		print("# of Record %d" %record_cnt)
		print("# of Field %d" %field_cnt)

# read file from command line argument
# argv[0] is this file name (csv_reader.py)
file = sys.argv[1]
read_dt(file) 