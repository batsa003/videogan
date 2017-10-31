import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from torch.autograd import Variable
import time

from model import Discriminator
from model import Generator
from data_loader import DataLoader
from logger import Logger
from utils import make_gif

logger = Logger('./logs')

num_epoch = 1
batchSize = 64
lr = 0.0002
    
discriminator = Discriminator()
generator = Generator()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

loss_function = nn.CrossEntropyLoss()
reg_loss_function = nn.L1Loss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, momentum = 0.5)
g_optim = torch.optim.Adam(generator.parameters(), lr=lr, momentum = 0.5)

dataloader = DataLoader()
data_size = dataloader.size
num_batch = data_size//batchSize
print('Total number of videos = ', data_size)
print('Total number of batches per echo = ', num_batch)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


start_time = time.time()
counter = 0
for epoch in range(num_epoch):
    for batch_index in range(num_batch): # [-1,32,
        videos = dataloader.get_batch().permute(0,2,1,3,4) # [64,3, 32, 64, 64]
        videos = to_variable(videos)
        real_labels = to_variable(torch.LongTensor(np.ones(batchSize, dtype = int)))
        fake_labels = to_variable(torch.LongTensor(np.zeros(batchSize, dtype = int)))

        discriminator.zero_grad()
        outputs = discriminator(videos)
        d_real_loss = loss_function(outputs, real_labels.long())
    
        real_score = outputs # Needed for tracking?

        noise = torch.rand(batchSize,100).view(-1,100,1,1,1)
        noise = to_variable(noise)

        fake_videos = generator(noise)
        outputs = discriminator(fake_videos)
        
        fake_score = outputs # Needed for tracking?
        
        d_fake_loss = loss_function(outputs, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optim.step()
        #print('Discriminator weights trained')
        
        #print("Training Generator..")
        noise = to_variable(torch.rand(batchSize,100).view(-1,100,1,1,1))

        generator.zero_grad()
        fake_videos = generator(noise)
        outputs = discriminator(fake_videos)
        g_loss = loss_function(outputs, real_labels.long())
        g_loss.backward()
        g_optim.step()

        info = {
            'd_fake_loss' : d_fake_loss.data[0],
            'd_real_loss' : d_real_loss.data[0],
            'g_loss' : g_loss.data[0],
            'real_score_mean' : real_score.data.mean(),
            'fake_score_mean' : fake_score.data.mean(),
        }
        for tag,value in info.items():
            logger.scalar_summary(tag, value, counter)

        if (batch_index)%50 == 0:
            print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %2.f, D(G(x)): %.2f, time: %4.4f"
		        %(epoch+1, num_epoch, batch_index+1, num_batch, d_loss.data[0], g_loss.data[0],
		        real_score.data.mean(), fake_score.data.mean(), time.time()-start_time))

        if batch_index % 200 == 0:
            if not os.path.isdir('videos/'):
                os.mkdir('videos')
            make_gif(denorm(fake_videos.data.cpu()[0]), 'videos/fake_gifs_%s_%s.gif' % (epoch, batch_index))
            print('Gif saved')

torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
print('Done')
