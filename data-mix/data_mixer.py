"""
Takes in positive seq files, negative sequence, 
pos features and negative features and creates four output files:
pos/neg shuffled, with labels with sequences
pos/neg same as above, without sequences
random should always be seeded with 1 to get consistent results
"""

import random
import pandas as pd
import argparse



def combine(sequence_file, features_file, label):
	""" outputs two pandas dataframes, both shuffled the same way.
		first has only sequences, the other also has features. 
		a label column is also added.
	"""
	shortname = sequence_file.split('.')[0]

	# read fasta file by making list of strings, each string is a sequence
	seqs = list()
	with open(sequence_file) as fastafile:
		cur_seq = ""
		for line in fastafile:
			if line[0] != '>':
				cur_seq += line	
			elif cur_seq != "":
				seqs.append(cur_seq.strip(' \t\"\n'))
				cur_seq = ""
	seqs = pd.DataFrame.from_dict({'sequence':seqs})
	feats = pd.read_csv(features_file, delim_whitespace=True, header=None, prefix='EDeN_')
	print("Processing ", shortname)
	print("Features shape: ", feats.shape)
	print("Sequence shape: ", seqs.shape)
	combined = pd.concat((feats, seqs), axis=1)
	combined['label'] =	label 
	# shuffle once
	combined = combined.sample(frac=1).reset_index(drop=True)
	# take out sequences for seperate dataframe
	no_seq = combined.drop('sequence', axis=1)

	# save to file
	combined.to_csv(shortname+'_randomized.csv', sep='\t', index=False)
	no_seq.to_csv(shortname+'_randomized_no_seq.csv', sep='\t', index=False)
	


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Combine and shuffle feature and sequence data. Also outputs version without sequences.\n Files are saved to same directory as sequence_file under the same name with extensions removed and \"_randomized.csv\" added.')

	parser.add_argument('features_file',
					help='file containing features')

	parser.add_argument('sequence_file',
					help='fasta file containing sequences')

	parser.add_argument('label',
					help='can be any integer, number of class. ie 0 for negative, 1 for positive.', type=int)

	args = parser.parse_args()

	combine(args.features_file, args.sequence_file, args.label)



