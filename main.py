import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.autograd import Variable
import time
import logging
from model import Discriminator
from model import Generator
from data_loader import DataLoader
from logger import Logger
from utils import make_gif
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Text Logger
def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('training_log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

# Saves [3, 64, 64] tensor x as image.
def save_img(x, filename): 
    x = denorm(x)
    x = x.squeeze()
    to_pil = ToPILImage()
    img = to_pil(x)
    img.save(filename)

def to_variable(x, requires_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad)

def denorm(x):
    out = (x + 1.0) / 2.0
    return nn.Tanh(out)

num_epoch = 5
batchSize = 64
lr = 0.0002
l1_lambda = 10

text_logger = setup_logger('Train')
logger = Logger('./logs')
    
discriminator = Discriminator()
generator = Generator()
discriminator.apply(weights_init)
generator.apply(weights_init)
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

loss_function = nn.CrossEntropyLoss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr, [0.5, 0.999])
g_optim = torch.optim.Adam(generator.parameters(), lr, [0.5, 0.999])

dataloader = DataLoader(batchSize)
data_size = len(dataloader.train_index)
num_batch = data_size//batchSize
#text_logger.info('Total number of videos for train = ' + str(data_size))
#text_logger.info('Total number of batches per echo = ' + str(num_batch))

start_time = time.time()
counter = 0
DIR_TO_SAVE = "./gen_videos/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)
sample_input = None
sample_input_set = False

for current_epoch in tqdm(range(1,num_epoch+1)):
    n_updates = 1
    for batch_index in range(num_batch):
        videos = dataloader.get_batch().permute(0,2,1,3,4) # [-1,3,32,64,64]
        videos = to_variable(videos)
        real_labels = to_variable(torch.LongTensor(np.ones(batchSize, dtype = int)), requires_grad = False)
        fake_labels = to_variable(torch.LongTensor(np.zeros(batchSize, dtype = int)), requires_grad = False)

        if not sample_input_set:
            sample_input = videos[0:1,:,0:1,:,:]
            sample_input_set = True

        if n_updates % 2 == 1:
            discriminator.zero_grad()
            generator.zero_grad()
            outputs = discriminator(videos).squeeze() # [-1,2]
            d_real_loss = loss_function(outputs, real_labels)
            fake_videos_d = generator(videos[:,:,0:1,:,:])
            outputs = discriminator(fake_videos_d).squeeze()
            d_fake_loss = loss_function(outputs, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()
            info = {
                 'd_loss': d_loss.data[0]
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)
        else:
            discriminator.zero_grad()
            generator.zero_grad()
            first_frame = videos[:,:,0:1,:,:]
            fake_videos = generator(first_frame)
            outputs = discriminator(fake_videos).squeeze()
            gen_first_frame = fake_videos[:,:,0:1,:,:]
            reg_loss = torch.mean(torch.abs(first_frame - gen_first_frame)) * l1_lambda
            g_loss = loss_function(outputs, real_labels) + reg_loss
            g_loss.backward()
            g_optim.step()
            info = {
                'g_loss' : g_loss.data[0],
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)

            '''
            # Calculate validation loss
            videos = to_variable(dataloader.get_batch('test').permute(0,2,1,3,4)) # [64,3, 32, 64, 64]
            first_frame = videos[:,:,0:1,:,:]
            fake_videos = generator(first_frame)
            outputs = discriminator(fake_videos).squeeze()
            gen_first_frame = fake_videos[:,:,0:1,:,:]
            err = torch.mean(torch.abs(first_frame - gen_first_frame)) * l1_lambda
            g_val_loss = loss_function(outputs, real_labels) + err
            info = {
                'g_val_loss' : g_val_loss.data[0],
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)
            '''

        n_updates += 1

        if (batch_index + 1) % 5 == 0:
            text_logger.info("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, \
                             g_val_loss: %.4f, time: %4.4f" \
                             % (current_epoch, num_epoch, batch_index+1, num_batch, \
                             d_loss.data[0], g_loss.data[0], g_val_loss.data[0], time.time()-start_time))

        counter += 1

        if (batch_index + 1) % 100 == 0:
            gen_out = generator(sample_input)

            save_img(sample_input.data.cpu(), DIR_TO_SAVE + 'fake_gifs_sample_%s_%s_a.jpg' % (current_epoch, batch_index))
            make_gif(denorm(gen_out.data.cpu()[0]), DIR_TO_SAVE + 'fake_gifs_sample__%s_%s_b.gif' % (current_epoch, batch_index))

            save_img(first_frame[0].data.cpu(), DIR_TO_SAVE + 'fake_gifs_%s_%s_a.jpg' % (current_epoch, batch_index))
            make_gif(denorm(fake_videos.data.cpu()[0]), DIR_TO_SAVE + 'fake_gifs_%s_%s_b.gif' % (current_epoch, batch_index))

            text_logger.info('Gifs saved at epoch: %d, batch_index: %d' % (current_epoch, batch_index))

        if (batch_index + 1) % 1000 == 0:
            torch.save(generator.state_dict(), './generator.pkl')
            torch.save(discriminator.state_dict(), './discriminator.pkl')
            text_logger.info('Saved the model to generator.pkl and discriminator.pkl')
            
        # Decay the learning rate
        if (batch_index + 1) % 1000 == 0:
            lr = lr / 10.0
            text_logger.info('Decayed learning rate to %.16f' % lr)
            for param_group in d_optim.param_groups:
                param_group['lr'] = lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = lr

torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
