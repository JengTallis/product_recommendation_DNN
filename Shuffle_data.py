import csv
import random


def shuffleData(file):
    with open(file, 'r') as r, open("shuffled.csv", 'w', newline='') as wr:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        out = csv.writer(wr, delimiter=",", quotechar='|')

        field, rdm = next(inp), []
        out.writerow(field)

        for row in inp:
            rdm.append(row)
        random.shuffle(rdm)

        for row in rdm:
            out.writerow(row)

def splitting ():
    i = ''
    mList = []
    print('How would you like to split the data?')
    print('Total sum of answer should be 100')
    print('recommended: 60, 10, 30')
    x = float(input('training:'))
    y = float(input('cross validation:'))
    z = float(input('test:'))
    while (x+y+z) != 100:
        print('Try again.')
        print('Total sum of answer should be 100')
        x = float(input('training:'))
        y = float(input('cross validation:'))
        z = float(input('test:'))
    split = [x/100, y/100, z/100]
    return split

def countRow(file):
    with open(file, 'r') as r:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        field = next(inp)
        rows = float(sum(1 for row in inp))
        print (rows)
        return rows





def splitData(file):
    split = splitting()
    with open(file, 'r') as r, open("training_set.csv", 'w', newline='') as wr1, open("cross_validation.csv", 'w', newline='') as wr2, open("test_set.csv", 'w', newline='') as wr3:
        inp = csv.reader(r, delimiter=",", quotechar='|')
        train = csv.writer(wr1, delimiter=",", quotechar='|')
        cross = csv.writer(wr2, delimiter=",", quotechar='|')
        test = csv.writer(wr3, delimiter=",", quotechar='|')

        field = next(inp)
        train.writerow(field)
        cross.writerow(field)
        test.writerow(field)

        rows = countRow(file)
        for i in range(len(split)):
            split[i] =(round((split[i] * rows),0))

        i = 0
        for row in inp:
            if i< int(split[0]):
                train.writerow(row)
            elif i < int((split[0]) + (split[1])):
                cross.writerow(row)
            else:
                test.writerow(row)
            i+=1

shuffleData('featurized.csv')
splitData('shuffled.csv')