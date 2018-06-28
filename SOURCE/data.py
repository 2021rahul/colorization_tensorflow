# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:27:28 2018

@author: rahul.ghosh
"""

import pandas as pd
import numpy as np
import cv2
import os
import config


class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        greyimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
        colorimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2LAB)
        return np.reshape(greyimg, (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), colorimg[:, :, 1:]

    def generate_batch(self):
        batch = []
        labels = []
        for i in range(self.batch_size):
            filename = os.path.join(config.DATA_DIR, self.dir_path, self.filelist[self.data_index])
            print(filename)
            greyimg, colorimg = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)
        labels = np.asarray(labels)/255.
        return batch, labels
