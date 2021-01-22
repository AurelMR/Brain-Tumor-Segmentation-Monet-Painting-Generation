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

def upsample(in_c, out_c, kernel_size=4, stride=2, padding=1,dropout=False):
    if dropout:
      conv = nn.Sequential(
          nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
          nn.InstanceNorm2d(out_c,affine=True),
          nn.Dropout(0.5),
          nn.ReLU()
      )
    else:
      conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_c,affine=True),
        nn.ReLU()
    )
    return conv

class Generator(nn.Module):
    def __init__(self, in_c=3):
        super(Generator, self).__init__()
        self.in_c = in_c

        self.down1 = downsample(self.in_c, 64, 4, 2,normalize=False) #(bs, 64, 128, 128)
        self.down2 = downsample(64, 128, 4, 2) #(bs, 128, 64, 64)
        self.down3 = downsample(128, 256, 4, 2) #(bs, 256, 32, 32)
        self.down4 = downsample(256, 512, 4, 2) #(bs, 512, 16, 16)
        self.down5 = downsample(512, 512, 4, 2) #(bs, 512, 8, 8)
        self.down6 = downsample(512, 512, 4, 2) #(bs, 512, 4, 4)
        self.down7 = downsample(512, 512, 4, 2) #(bs, 512, 2, 2)
        self.down8 = downsample(512, 512, 4, 2) #(bs, 512, 1, 1)

        self.up8 = upsample(512, 512, 4, 2,dropout=True) #(bs, 512, 2, 2)
        self.up7 = upsample(1024, 512, 4, 2,dropout=True) #(bs, 512, 4, 4)
        self.up6 = upsample(1024, 512, 4, 2,dropout=True) #(bs, 512, 8, 8)
        self.up5 = upsample(1024, 512, 4, 2) #(bs, 512, 16, 16)
        self.up4 = upsample(1024, 256, 4, 2) #(bs, 256, 32, 32)
        self.up3 = upsample(512, 128, 4, 2) #(bs, 128, 64, 64)
        self.up2 = upsample(256, 64, 4, 2) #(bs, 64, 128, 128)

  
        self.up1 = nn.Sequential(
                      nn.ConvTranspose2d(128,3,4,2,1), #upsample without ReLU
                      nn.Tanh()
                      )
        
    def forward(self,input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        y7 = torch.cat((self.up8(x8),x7),axis=1) #(bs,1024,2,2)
        y6 = torch.cat((self.up7(y7),x6),axis=1) #(bs,1024,4,4)
        y5 = torch.cat((self.up6(y6),x5),axis=1) #(bs,1024,8,8)
        y4 = torch.cat((self.up5(y5),x4),axis=1) #(bs,1024,16,16)
        y3 = torch.cat((self.up4(y4),x3),axis=1) #(bs,512,32,32)
        y2 = torch.cat((self.up3(y3),x2),axis=1) #(bs,256,64,64)
        y1 = torch.cat((self.up2(y2),x1),axis=1) #(bs,128,128,128)

        return self.up1(y1)
