from __future__ import print_function
"""
This code will take input csv and convert it to a vector
for input in a net. Then the correctly shaped net will train
on input output pairs generated in batches, randomly
finally, the network is verified on some witheld data.

Make sure that the path for train_model point to whitespace delimited files that you want
"""
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import losses, layers

import pandas as pd
import numpy as np
import argparse
import pickle
import pdb

def train_model(train, validation, test):
	""" 
	Train on train, and test on test
	"""
	train_data = train[:,:-1]
	train_labels = train[:,-1]
	validation_data = validation[:,:-1]
	validation_labels = validation[:,-1]
	test_data = test[:,:-1]
	test_labels = test[:,-1]
	# num inputs is the number of columns in the csv excluding the last one
	num_inputs = train_data.shape[1]
	print(num_inputs)
	print(train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape, test_data.shape, test_labels.shape) 
	print(max(train_labels))

	to_onehot = lambda labels : np.eye(2)[labels.astype(int)]
	train_labels = to_onehot(train_labels)
	validation_labels = to_onehot(validation_labels)
	test_labels = to_onehot(test_labels)

	print(train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape, test_data.shape, test_labels.shape) 
	# create and compile the model
	model = Sequential()

	model.add(keras.layers.normalization.BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one", weights=None, batch_input_shape=(None, num_inputs), gamma_regularizer=None, momentum=0.99, axis=-1))


	model.add(Dense(1000, input_dim=num_inputs))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(900))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(100, name='learned_features'))
	model.add(Activation('relu'))

	model.add(Dense(2))
	model.add(Activation('softmax'))

	rms = RMSprop()
	loss = losses.categorical_crossentropy 
	model.compile(loss=loss, optimizer=rms, metrics=['accuracy'])

	# train the model
	model.fit(train_data, train_labels, epochs=40, batch_size=100, validation_data=(test_data, test_labels), shuffle=True)

	# test the model for overfitting using withheld test data
	scores = model.evaluate(train_data, train_labels)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	return model


#open pos and neg files
pos_features = pd.read_csv('../informatics-data/pos_features_shuffled_noseq.csv', delim_whitespace=True)
neg_features = pd.read_csv('../informatics-data/neg_features_shuffled_noseq.csv', delim_whitespace=True)

#drop sequences, we don't need them
#pos_features = pos_features.drop('sequence', axis=1)
#neg_features = neg_features.drop('sequence', axis=1)
neg_features = neg_features.drop('index', axis=1)

#add label column
pos_features['label'] = np.ones((pos_features.shape[0]))
neg_features['label'] = np.zeros((neg_features.shape[0]))

#divide into 80% training data, 5% verification data, and 10% testing data
num_pos_features = pos_features.shape[0]
num_neg_features = neg_features.shape[0]  

pos_train = pos_features.loc[:int(num_pos_features*0.8)].as_matrix()
pos_validation = pos_features.loc[int(num_pos_features*0.8):int(num_pos_features*0.85)].as_matrix()
pos_test = pos_features.loc[int(num_pos_features*0.85):].as_matrix()
neg_train= neg_features.loc[:int(num_neg_features*0.8)].as_matrix()
neg_validation = neg_features.loc[int(num_neg_features*0.8):int(num_neg_features*0.85)].as_matrix()
neg_test = neg_features.loc[int(num_neg_features*0.85):].as_matrix()

train = np.concatenate((pos_train, neg_train))
validation = np.concatenate((pos_validation, neg_validation))
test = np.concatenate((pos_test, neg_test))


model = train_model(train, validation, test)
model.save('models/single_model1.h5')
