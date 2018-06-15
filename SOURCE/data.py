# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:27:28 2018

@author: rahul.ghosh
"""

import pandas as pd
import numpy as np
import os
import config


class DATA():

    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.data_index = 0
        self.dataX = None
        self.dataY = None
        self.size = None

    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename), header=None)
        data = np.asarray(data)
        self.size, self.num_features = data.shape
        self.dataX = data[:, :-1]
        self.dataY = self.dense_to_one_hot(data[:, -1])

    def dense_to_one_hot(self, labels):
        labels_one_hot = pd.get_dummies(labels)
        return np.asarray(labels_one_hot, dtype = np.float)

    def generate_batch(self):
        batch = np.ndarray(shape=(config.BATCH_SIZE), dtype=np.float32)
        labels = np.ndarray(shape=(config.BATCH_SIZE, 1), dtype=np.float32)
        batch = self.dataX[self.data_index:self.data_index+self.batch_size,:]
        labels = self.dataY[self.data_index:self.data_index+self.batch_size, :]
        self.data_index = (self.data_index + self.batch_size) % self.size
        return batch, labels
