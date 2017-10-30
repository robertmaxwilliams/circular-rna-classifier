import pandas
import numpy as np
import pickle
import argparse


def pickle_data(filename, output_filename):
	""" 
	Pickle training data found at filename, as list of two numpy array
	"""
	# import the csv as numpy array
	data = pandas.read_csv(filename, delim_whitespace=True, lineterminator='\n').values
	# shuffle the values in place, since data is clumped by class
	np.random.shuffle(data)
	
	# get all but last column for 'X', input data
	X = data[:, :-1] 
	X = X.astype(float)

	# get only the last row for 'y', targets and convert to 2-wide onehot
	y = data[:, -1]
	y = y.astype(int)	
	y_onehot = np.zeros((y.shape[0], 2))
	y_onehot[np.arange(y.shape[0]), y] = 1.
	y = y_onehot
	
	with open(output_filename, 'w') as outfile:
		pickle.dump([X, y], outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('train_file',
            help='a csv file, tab delimited, where all entries are numeric and the last is the training label',
            )
    parser.add_argument('output_filename',
            help='name of file to output, should end in .pickle',
			default='data.pickle'
            )
	args = parser.parseargs()
	pickle_data(


