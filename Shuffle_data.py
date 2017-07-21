from random import shuffle
fid = open('final.csv','r')
data = fid.readlines()
header, rest = data[0], data[1:]
shuffle(rest)

with open('output.csv', 'w') as out:
	out.write(''.join([header] + rest))