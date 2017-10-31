import torch
from ops import *
from torch.autograd import Variable

class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                deconv2d_first(100,512), #[-1,512,4,4]
                batchNorm4d(512),
                relu(),
                deconv2d(512,256),
                batchNorm4d(256),
                relu(),
                deconv2d(256,128),
                batchNorm4d(128),
                relu(),
                deconv2d(128,64),
                batchNorm4d(64),
                relu(),
                deconv2d(64,3),
                nn.Tanh()
                )
    def forward(self,x):
        out = self.model(x)
        return out

class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
                deconv3d_first(100,512), #[-1,512,4,4]
                batchNorm5d(512),
                relu(),
                deconv3d(512,256),
                batchNorm5d(256),
                relu(),
                deconv3d(256,128),
                batchNorm5d(128),
                relu(),
                deconv3d(128,64),
                batchNorm5d(64),
                relu()
                )
    def forward(self,x):
        out = self.model(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.background = G_background()
        self.video = G_video()
        self.mymodules = nn.ModuleList([
            nn.Sequential(deconv3d(64,3), nn.Tanh()),
            nn.Sequential(deconv3d(64,1), nn.Sigmoid()) # TODO: Add L1 Penalty.
        ])

    def forward(self,x):
        x = x.view((-1,100,1,1,1)) #[-1,100,1,1,1]
        video = self.video(x) #[-1, 64, 16, 32, 32], which will be used for generating the mask and the foreground

        gen_net = self.mymodules[0]
        foreground = gen_net(video) #[-1,3,32,64,64]
        
#        mask_net = nn.Sequential(deconv3d(64,1), nn.Sigmoid()) # TODO: Add L1 Penalty.
        mask_net = self.mymodules[1]
        mask = mask_net(video) #[-1,1,32,64,64]
        mask_repeated = mask.repeat(1,3,1,1,1) # repeat for each color channel. [-1, 3, 32, 64, 64]
        
        x = x.view((-1,100,1,1))
        background = self.background(x) # [-1,3,64,64]
        background_frames = background.unsqueeze(2).repeat(1,1,32,1,1) # [-1,3,32,64,64]
        out = torch.mul(mask_repeated,foreground) + torch.mul(1-mask_repeated, background_frames)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                conv3d(3, 64), #[-1, 64, 16, 32, 32]
                lrelu(0.2), 
                conv3d(64,128), #[-1, 126,8,16,16]
                batchNorm5d(128, 1e-3), 
                lrelu(0.2),
                conv3d(128,256), #[-1,256,4,8,8]
                batchNorm5d(256, 1e-3),
                lrelu(0.2),
                conv3d(256,512), #[-1,512,2,4,4]
                batchNorm5d(512,1e-3),
                lrelu(0.2),
                conv3d(512,2, (2,4,4), (1,1,1), (0,0,0)) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        
    def forward(self, x):
        out = self.model(x).squeeze()
        return out

if __name__ == '__main__':
    for i in range(2):
        x = Variable(torch.rand([20, 3, 32, 64, 64]).cuda())
        model = Discriminator().cuda()
        print('Discriminator input', x.size())
        out = model(x)
        print('Discriminator out ', out.size())

        x = Variable(torch.rand([20,100]).cuda())
        x = x.view((-1,100,1,1,1))
        print('Generator input', x.size())
        model = Generator().cuda()
        out = model(x)  
        print('Generator out ', out.size())
        print(type(out.data[0]))
        print(out.data[0].size())
