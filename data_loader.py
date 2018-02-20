import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os
import cv2 
import numpy as np

# Golf video dataset stabilized by SIFT + RANSAC.
# Downloaded from http://data.csail.mit.edu/videogan/golf.tar.bz2
# Extracting this gives you frames-stable-many folder.
# Also, download the file listings from here: http://data.csail.mit.edu/videogan/golf.txt

GOLF_DATA_LISTING = '/srv/bat/data/frames-stable-many/golf.txt'
DATA_ROOT = '/srv/bat/data/frames-stable-many/'

class DataLoader(object):

    def __init__(self, batch_size = 5): 
        #reading data list
        self.batch_size = batch_size
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128 
        self.train = None
        self.test = None
    
        # Shuffle video index.
        data_list_path = os.path.join(GOLF_DATA_LISTING) #603776 video path
        with open(data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self.video_index)

        self.size = len(self.video_index)
        self.train_index = self.video_index[:self.size//2]
        self.test_index = self.video_index[self.size//2:]

		# A pointer in the dataset
        self.cursor = 0

    def get_batch(self, type_dataset='train'):
        if type_dataset not in('train', 'test'):
            print 'type_dataset = ', type_dataset, ' is invalid. Returning None'
            return None

        dataset_index = self.train_index if type_dataset == 'train' else self.test_index
        if self.cursor + self.batch_size > len(dataset_index):
            self.cursor = 0
            np.random.shuffle(dataset_index)

        t_out = torch.zeros((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.

        for idx in xrange(self.batch_size):
            video_path = os.path.join(DATA_ROOT, dataset_index[self.cursor])
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self.image_size
            for j in xrange(self.frame_size):
                if j < count:
                    cut = j * self.image_size
                else:
                    cut = (count - 1) * self.image_size
                crop = inputimage[cut : cut + self.image_size, :]
                temp_out = to_tensor(cv2.resize(crop, (self.crop_size, self.crop_size)))
                temp_out = temp_out * 2 - 1
#                for cc in range(3):
#                    temp_out[cc,:,:] -= temp_out[cc,:,:].mean() # (According to Line 123 in donkey_video2.lua)
#                t_out[idx,j,:,:,:] = temp_out

            self.cursor += 1

        return t_out

d = DataLoader().get_batch('test')
