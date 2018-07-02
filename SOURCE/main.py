# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:00:55 2018

@author: rahul.ghosh
"""

import data
import model
import config

if __name__ == "__main__":
    # READ DATA
    train_data = data.DATA(config.TRAIN_DIR)
    # BUILD MODEL
    model = model.MODEL()
    model.build()
    # TRAIN MODEL
    model.train(train_data)
    # TEST MODEL
    test_data = data.DATA(config.TEST_DIR)
    model.test(test_data)
