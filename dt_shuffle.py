
''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
shuffle.py

Shuffle csv file

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from random import shuffle

rf = open('data.csv','r')
data = rf.readlines()
header, rest = data[0], data[1:]
shuffle(rest)

with open('shuffle.csv', 'w') as wf:
	wf.write(''.join([header] + rest))