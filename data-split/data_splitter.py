from __future__ import print_function
"""
Divides the data into size files:
training_pos.csv 	with 0.8 of data points 
validation_pos.csv 	with 0.05 of data points
testing_pos.csv 	with 0.15 of data points
training_neg.csv	""
validation_neg.csv	""
testing_neg.csv  	""
"""
import random

positive_lines = list()
negative_lines = list()
with open('all_features.csv', 'r') as infile:
	for line in infile:#note: lines have newline on them!
		if line[-3] == '0':
			negative_lines.append(line[:-4])
		if line[-3] == '1':
			positive_lines.append(line[:-4])

pos_size = len(positive_lines)
training_pos = positive_lines[0 : int(pos_size*0.8)]
validation_pos = positive_lines[int(pos_size*0.8) : int(pos_size*0.85)]
testing_pos = positive_lines[int(pos_size*0.85) : ]

neg_size = len(negative_lines)
training_neg = negative_lines[0 : int(neg_size*0.8)]
validation_neg = negative_lines[int(neg_size*0.8) : int(neg_size*0.85)]
testing_neg = negative_lines[int(neg_size*0.85) : ]

def write_list(filename, listt):
	with open(filename, 'w') as outfile:
		for line in listt:
			outfile.write(line + '\n')

write_list('training_pos.csv', training_pos)  
write_list('validation_pos.csv', validation_pos)
write_list('testing_pos.csv', testing_pos)

write_list('training_neg.csv', training_neg)
write_list('validation_neg.csv', validation_neg)
write_list('testing_neg.csv', testing_neg)
