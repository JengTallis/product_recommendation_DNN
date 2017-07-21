''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt_add_features.py

Introduce new features for products and condense the n_months

1. Pair index pattern count over the n_months months (0,0), (0,1), (1,0), (1,1)
2. Longest same-index chain length (chain length of consecutive 0s), (chain length of consecutive 1)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
''' 

import csv


''' 
Chunk generator 
Take a CSV reader and yield chunksize sized slices. 
'''
def generate_chunk(reader, chunksize):
	chunk = []
	for index, line in enumerate(reader):
		if (index % chunksize == 0 and index > 0):
			yield chunk
			del chunk[:]
		chunk.append(line)
	yield chunk


'''
Condense Customer information and add new features:
Pair index pattern count 
Longest same-index chain length
'''
def add_features_dt(data):

	csv.field_size_limit(sys.maxsize)

	# open read_file and write_file
	with open(data, 'r', newline='') as rf, open ("features.csv", 'w', newline='') as wf: 
		reader = csv.reader(rf, skipinitialspace=True, delimiter=',', quotechar='|')	# csv_reader
		writer = csv.writer(wf, delimiter=",", quotechar='|')							# csv_writer

		n_products = 24

		fields = next(reader, None)	# read headers
		for i in range(n_products):
			features = ["(00)_%d" %i] + ["(01)_%d" %i] + ["(10)_%d" %i] + ["(11)_%d" %i] + ["(0s)_%d" %i] + ["(1s)_%d" %i] 
			fields = fields + features		
		#print(fields)
		writer.writerow(fields)		# write headers

		n_months = 17
		pb_idx = 18
		pe_idx = 18 + n_products

		for chunk in generate_chunk(reader, n_months):	# a chunk for each customer

			# create features for each product
			for p in range(pb_idx,pe_idx):	# each product

				pair_count = [0] * 4	# (0,0), (0,1), (1,0), (1,1)  
				chain_len = [0] * 2	# (0s), (1s)
				prev = -1
				zl = 0	# len(0s)
				ol = 0	# len(1s)

				for r in range(n_months-1): # scan monthly data

					# idx pattern count
					if r+2 < n_months: # control range
						pair_count[2*int(chunk[r][p]) + int(chunk[r+1][p])] += 1  # (a,b), 2a+b => 0,1,2,3

					# chian length
					if (prev == -1):	# head month
						if int(chunk[r][p]) < 1:
							zl += 1
						else:
							ol += 1
					elif (int(chunk[r][p]) == 0) and (prev == 0):	# continue 0
						zl += 1
					elif (int(chunk[r][p]) == 1) and (prev == 1):	# continue 1
						ol += 1
					elif (prev != -1):	# change 
						if (chain_len[0]) < zl: chain_len[0] = zl 
						if chain_len[1] < ol: chain_len[1] = ol 
						zl = 0
						ol = 0	
					prev = int(chunk[r][p])
				if (chain_len[0]) < zl: chain_len[0] = zl 
				if chain_len[1] < ol: chain_len[1] = ol 
				#print(chain_len)
				#print("zl and ol are %d, %d" %(zl, ol))
				chunk[n_months-1] = chunk[n_months-1] + pair_count + chain_len

			writer.writerow(chunk[n_months-1])

add_features_dt("sort.csv")
