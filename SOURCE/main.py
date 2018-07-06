# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:00:55 2018

@author: rahul.ghosh
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import data
import model
import config
import datetime

if __name__ == "__main__":
    with open(os.path.join(config.LOG_DIR, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")
        log.write("Use Pretrained Weights: " + str(config.USE_PRETRAINED) + "\n")
        log.write("Pretrained Model: " + config.PRETRAINED + "\n")
        # READ DATA
        train_data = data.DATA(config.TRAIN_DIR)
        print("Train Data Loaded")
        # BUILD MODEL
        model = model.MODEL()
        print("Model Initialized")
        model.build()
        print("Model Built")
        # TRAIN MODEL
        model.train(train_data, log)
        print("Model Trained")
        # TEST MODEL
        test_data = data.DATA(config.TEST_DIR)
        print("Test Data Loaded")
        model.test(test_data, log)
        print("Image Reconstruction Done")
