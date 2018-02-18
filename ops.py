import torch.nn as nn

def conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def deconv2d_first(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (4,4))

def deconv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def deconv3d_first(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,4,4))

def deconv3d_video(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,1,1))

def deconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def batchNorm4d(num_features, eps = 1e-5): #input: N, C, H, W
    return nn.BatchNorm2d(num_features, eps = eps)

def batchNorm5d(num_features, eps = 1e-5): #input: N, C, D, H, W
    return nn.BatchNorm3d(num_features, eps = eps)

def relu(inplace = True):
    return nn.ReLU(inplace)

def lrelu(negative_slope = 0.2, inplace = True):
    return nn.LeakyReLU(negative_slope, inplace)
