import torch
from ops import *
from torch.autograd import Variable
import os

class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                deconv2d(1024,512), #[-1,512,4,4]
                batchNorm4d(512),
                relu(),
                deconv2d(512,256),
                batchNorm4d(256),
                relu(),
                deconv2d(256,128),
                batchNorm4d(128),
                relu(),
                deconv2d(128,3),
                nn.Tanh()
                )

    def forward(self,x):
        #print('G_background Input =', x.size())
        out = self.model(x)
        #print('G_background Output =', out.size())
        return out

class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
                deconv3d_video(1024,1024), #[-1,512,4,4]
                batchNorm5d(1024),
                relu(),
                deconv3d(1024,512),
                batchNorm5d(512),
                relu(),
                deconv3d(512,256),
                batchNorm5d(256),
                relu(),
                deconv3d(256,128),
                batchNorm5d(128),
                relu(),
                )
    def forward(self,x):
        #print('G_video input =', x.size())
        out = self.model(x)
        #print('G_video output =', out.size())
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode = G_encode()
        self.background = G_background()
        self.video = G_video()
        self.gen_net = nn.Sequential(deconv3d(128,3), nn.Tanh())
        self.mask_net = nn.Sequential(deconv3d(128,1), nn.Sigmoid())

    def forward(self,x):
        #print('Generator input = ',x.size())
        x = x.squeeze(2)
        encoded = self.encode(x)
        encoded = encoded.unsqueeze(2)
        video = self.video(encoded) #[-1, 128, 16, 32, 32], which will be used for generating the mask and the foreground
        #print('Video size = ', video.size())

        foreground = self.gen_net(video) #[-1,3,32,64,64]
        #print('Foreground size =', foreground.size())
        
        mask = self.mask_net(video) #[-1,1,32,64,64]
        #print('Mask size = ', mask.size())
        mask_repeated = mask.repeat(1,3,1,1,1) # repeat for each color channel. [-1, 3, 32, 64, 64]
        #print('Mask repeated size = ', mask_repeated.size())
        
        x = encoded.view((-1,1024,4,4))
        background = self.background(x) # [-1,3,64,64]
        #print('Background size = ', background.size())
        background_frames = background.unsqueeze(2).repeat(1,1,32,1,1) # [-1,3,32,64,64]
        out = torch.mul(mask,foreground) + torch.mul(1-mask, background_frames)
        #print('Generator out = ', out.size())        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                conv3d(3, 128), #[-1, 64, 16, 32, 32]
                lrelu(0.2), 
                conv3d(128,256), #[-1, 126,8,16,16]
                batchNorm5d(256, 1e-3), 
                lrelu(0.2),
                conv3d(256,512), #[-1,256,4,8,8]
                batchNorm5d(512, 1e-3),
                lrelu(0.2),
                conv3d(512,1024), #[-1,512,2,4,4]
                batchNorm5d(1024,1e-3),
                lrelu(0.2),
                conv3d(1024,2, (2,4,4), (1,1,1), (0,0,0)) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        #self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])
        
    def forward(self, x):
        out = self.model(x).squeeze()
        #out = self.mymodules[0](out)
        return out

class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        self.model = nn.Sequential(
                conv2d(3,128),
                relu(),
                conv2d(128,256),
                batchNorm4d(256),
                relu(),
                conv2d(256,512),
                batchNorm4d(512),
                relu(),
                conv2d(512,1024),
                batchNorm4d(1024),
                relu(),
                )
    def forward(self,x):
        #print('G_encode Input =', x.size())
        out = self.model(x)
        #print('G_encode Output =', out.size())
        return out
'''
if __name__ == '__main__':
    for i in range(1):
        x = Variable(torch.rand([20, 3, 32, 64, 64]).cuda())
        model = Discriminator().cuda()
        print('Discriminator input', x.size())
        out = model(x).squeeze()
        print('Discriminator out ', out.size())

        x = Variable(torch.rand([20,3,1,64,64]).cuda())
        print('Generator input', x.size())
        model = Generator().cuda()
        out = model(x)  
        print('Generator out ', out.size())
        print(type(out.data[0]))
        print(out.data[0].size())
        x = Variable(torch.rand([13,3,64,64])).cuda()
        #x = Variable(torch.rand([13,3,1,64,64]))
        print('Generator input', x.size())
        model = Generator().cuda()
        out = model(x)  
        print('Generator out ', out.size())
'''
