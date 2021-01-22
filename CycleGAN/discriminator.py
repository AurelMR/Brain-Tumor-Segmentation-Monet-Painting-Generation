import torch
import torch.nn as nn
from torch.nn.functional import softmax

def downsample(in_c, out_c, kernel_size=4, stride=2, padding=1,normalize=True):
    if normalize:
      conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_c,affine=True),
        nn.LeakyReLU()
      )
    else:
      conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.LeakyReLU()
      )
    return conv


class Discriminator(nn.Module):
    def __init__(self, in_c=3):
        super(Discriminator, self).__init__()
        self.in_c = in_c

        self.gen = nn.Sequential(
            downsample(self.in_c, 64, 4, 2,False), #(bs, 64, 128, 128)
            downsample(64, 128, 4, 2), #(bs, 128, 64, 64)
            downsample(128, 256, 4, 2), #(bs, 256, 32, 32)

            nn.ZeroPad2d(1),
            nn.Conv2d(256,512,4,1), #(bs, 512, 31, 31)
            nn.InstanceNorm2d(512), #(bs, 512, 31, 31)
            nn.LeakyReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(512,1,4,1) #(bs, 1, 30, 30)
        )

    def forward(self,input):
        return self.gen(input)
