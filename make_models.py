from __future__ import print_function
"""
This code will take input csv and convert it to a vector
for input in a net. Then the correctly shaped net will train
on input output pairs generated in batches, randomly
finally, the network is verified on some witheld data.

Make sure that the path for train_model point to whitespace delimited files that you want
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import losses, layers

import pandas
import numpy as np
import argparse
import pickle

from tensorflow.python.lib.io import file_io

def train_model(data_file, epochs):
	""" 
	Train model on X and y, return model
	"""
	table = pandas.read_table(data_file)
	table = table.as_matrix()#actually numpy array
	data = table[:,:-1]
	labels = table[:,-1]
	data.

	table
	#create test data to check for overfitting
	X_test = X[:200]
	X = X[200:]	
	y_test = y[:200]
	y = y[200:]
	
	# num inputs is the number of columns in the csv excluding the last one
	num_inputs = X.shape[1]
	print('X shape: ', X.shape)
	print('y shape: ', y.shape)

	# create and compile the model
	model = Sequential()

	model.add(Dense(500, input_dim=num_inputs))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(300))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(10, name='learned_features'))
	model.add(Activation('relu'))

	model.add(Dense(2))
	model.add(Activation('softmax'))

	rms = RMSprop()
	loss = losses.categorical_crossentropy 
	model.compile(loss=loss, optimizer=rms, metrics=['accuracy'])

	# train the model
	model.fit(X, y, epochs=50, batch_size=100)

	# test the model for overfitting using withheld test data
	scores = model.evaluate(X_test, y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	return model


def train_model_cloud(train_file, **args):
	# Here put all the main training code in this function
	print(train_file)
	file_stream = file_io.FileIO(train_file, mode='r')
	X, y = pickle.load(file_stream)
	model = train_model(X, y)
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Input Arguments
	parser.add_argument('train_file',
			help='a csv file, tab delimited, where all entries are numeric and the last is the training label',
			)
	parser.add_argument('out_file',
			help='name of file to output, should end in .h5'
			)
	parser.add_argument('--epochs',
			help='number of epochs to train',
			default=20,
			type=int
			)

	args = parser.parse_args()

	model = train_model(args.train_file, args.epochs)
	model.save(args.out_file)
