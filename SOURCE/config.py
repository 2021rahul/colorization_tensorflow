# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:16:29 2018

@author: rahul.ghosh
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'TensorFlow/RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'TensorFlow/MODEL/')

TRAIN_FILENAME = "train_iris.csv"
TEST_FILENAME = "test_iris.csv"

# DATA INFORMATION
NUM_FEATURES = 4
NUM_CLASS = 3
BATCH_SIZE = 45

# MODEL INFORMATION
HIDDEN_LAYERS = 2
SHAPE = [(NUM_FEATURES, 8),
         (8, 4),
         (4, NUM_CLASS)]

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
NUM_EPOCHS = 2000