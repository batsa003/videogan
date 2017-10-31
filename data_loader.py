import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

class DataLoader(object):

    def __init__(self):
        #reading data list
        self.data_root = '/home/bat/temp/frames-stable-many/'
        self.batch_size = 64
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128
        self.data_list_path = os.path.join(self.data_root, 'golf.txt') #603776 video path
        with open(self.data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self.video_index)
        self.size = len(self.video_index)
        self.cursor = 0

    def get_batch(self):
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.video_index)

        to_tensor = transforms.ToTensor()
        t_out = torch.zeros((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))

        for idx in xrange(self.batch_size):
            video_path = os.path.join(self.data_root, self.video_index[self.cursor])
            self.cursor += 1
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self.image_size
            for j in xrange(self.frame_size):
                if j < count:
                    cut = j * self.image_size
                else:
                    cut = (count - 1) * self.image_size
                crop = inputimage[cut : cut + self.image_size, :]
                t_out[idx,j,:,:,:] = to_tensor(cv2.resize(crop, (self.crop_size, self.crop_size)))

        return t_out

if __name__ == '__main__':
	# Tested on notebook. Haven't executed it from terminal.
	dataloader = DataLoader()
	print('returned from get_batch() :', dataloader.get_batch().size())
	imgs = dataloader.get_batch()[0]

	t1 = imgs[0] #[3,64,64]
	t2 = imgs[7]
	t3 = imgs[15]
	t4 = imgs[31]
	o1 = np.transpose(t1.numpy(), (1,2,0)) #[64,64,3]
	o2 = np.transpose(t2.numpy(), (1,2,0))
	o3 = np.transpose(t3.numpy(), (1,2,0))
	o4 = np.transpose(t4.numpy(), (1,2,0))

	print(o1.mean())
	print(o1.max())
	print(o1.min())
	print(o1.std())

	plt.figure()
	plt.imshow(o1) # Frame 1
	plt.figure()
	plt.imshow(o2) # Frame 8
	plt.figure()
	plt.imshow(o3) # Frame 16
	plt.figure()
	plt.imshow(o4) # Frame 32
