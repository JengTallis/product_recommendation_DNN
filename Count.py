import pandas as pd

for x in range(504367):
	csvin = pd.read_csv('complete.csv', skiprows = x*17 + 1, nrows = 16)
	prev_val = 0

	count = [[0 for i in range(4)] for j in range(24)]

	for j in range(18,42):
		row_iterator = csvin.iterrows()
		_, last = row_iterator.__next__()

		for index, row in row_iterator: 
			prev_val = float(last[j])
			data = float(row[j]) - prev_val

			if (data == 0):
				if (prev_val == 1):
					count[j-18][2] += 1
				else:
					count[j-18][0] += 1 
			elif (data == 1):
				count[j-18][1] += 1
			elif (data == -1):
				count[j-18][3] += 1
		
			last = row

	print (count)
