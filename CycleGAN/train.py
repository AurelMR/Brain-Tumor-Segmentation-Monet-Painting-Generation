import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import os
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import ImageLoader
from discriminator import Discriminator
from generator import Generator


# IMG_ROOT = '../Scripts/dataset/train_image/'
# LABEL_ROOT = '../Scripts/dataset/train_label/'

BASE_PATH='/gpfs/workdir/houdberta/gan-getting-started/'
MONET_PATH = os.path.join(BASE_PATH, 'monet_jpg')
PHOTO_PATH = os.path.join(BASE_PATH, 'photo_jpg')

BATCH_SIZE = 4
trainset = ImageLoader(MONET_PATH,PHOTO_PATH)
loader = DataLoader(trainset,BATCH_SIZE,shuffle=True)

device = 'cuda:0'

def discriminator_real_loss(real):
  criterion = nn.BCEWithLogitsLoss()
  real_loss = criterion(real,torch.ones_like(real))
  return real_loss
  
def discriminator_fake_loss(generated):
  criterion = nn.BCEWithLogitsLoss()
  generated_loss = criterion(generated,torch.zeros_like(generated))
  return generated_loss
  
def id_loss(real,generated,Lambda=2e-4):
  return Lambda * torch.mean(torch.absolute(real - generated))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
monet_generator = Generator().to(device)
photo_generator = Generator().to(device)
monet_discriminator = Discriminator().to(device)
photo_discriminator = Discriminator().to(device)

monet_generator = monet_generator.apply(weights_init)
photo_generator = photo_generator.apply(weights_init)
monet_discriminator = monet_discriminator.apply(weights_init)
photo_discriminator = photo_discriminator.apply(weights_init)


n_epoch = 50
BATCH_SIZE = 8
LAMBDA=10
lr = 2e-4
save = True

trainset = ImageLoader(MONET_PATH,PHOTO_PATH)
loader = DataLoader(trainset,BATCH_SIZE,shuffle=True)
len_loader = len(loader)
i=0

monet_generator.train()
photo_generator.train()
monet_discriminator.train()
photo_discriminator.train()

generator_optimizer = torch.optim.Adam(itertools.chain(monet_generator.parameters(),photo_generator.parameters()),lr,betas=(0.5,0.999))
monet_generator_optimizer = torch.optim.Adam(monet_generator.parameters(),lr,betas=(0.5,0.999))
photo_generator_optimizer = torch.optim.Adam(photo_generator.parameters(),lr,betas=(0.5,0.999))
monet_discriminator_optimizer = torch.optim.Adam(monet_discriminator.parameters(),lr,betas=(0.5,0.999))
photo_discriminator_optimizer = torch.optim.Adam(photo_discriminator.parameters(),lr,betas=(0.5,0.999))


fake_monet_gen_loss_history = []
fake_photo_gen_loss_history = []
cycled_monet_gen_loss_history = []
cycled_photo_gen_loss_history = []
same_monet_gen_loss_history = []
same_photo_gen_loss_history = []
real_monet_discr_loss_history = []
real_photo_discr_loss_history = []

photoNames=['22fc4c2e5b','385d53e164','5f7a943ad3','e6a609cf76','dc2abc2e66']
monetNames=['bf6db09354','26b66eb819','b76d52e05a','1078363ff0','a96b79a93f']
savePath='/gpfs/workdir/houdberta/monet_per_epoch_images/'
try:
    os.mkdir(savePath)
except:
    pass

for epoch in range(n_epoch):

  for real_monet,real_photo,_,_ in loader:

    generator_optimizer.zero_grad()

    #--------------------------------------
    #--------  Steps on generators --------
    #--------------------------------------

    real_monet,real_photo = real_monet.to(device),real_photo.to(device)

    #Generating sames and computing same losses
    same_monet = monet_generator(real_monet)
    same_photo = photo_generator(real_photo)
    same_monet_gen_loss = id_loss(real_monet,same_monet,Lambda=0.5*LAMBDA)
    same_photo_gen_loss = id_loss(real_photo,same_photo,Lambda=0.5*LAMBDA)

    #Generating fake photos and monets
    fake_monet = monet_generator(real_photo)
    fake_monet_discr = monet_discriminator(fake_monet)
    fake_monet_discr_loss = discriminator_real_loss(fake_monet_discr)

    fake_photo = photo_generator(real_monet)
    fake_photo_discr = photo_discriminator(fake_photo)
    fake_photo_discr_loss = discriminator_real_loss(fake_photo_discr)

    #Generating fakes from fakes
    cycled_monet = monet_generator(fake_photo)
    cycled_monet_gen_loss = id_loss(real_monet,cycled_monet,Lambda=LAMBDA)
    cycled_photo = photo_generator(fake_monet)
    cycled_photo_gen_loss = id_loss(real_photo,cycled_photo,Lambda=LAMBDA)

    gen_loss = fake_monet_discr_loss + fake_photo_discr_loss + cycled_monet_gen_loss + cycled_photo_gen_loss + same_monet_gen_loss + same_photo_gen_loss
    gen_loss.backward()
    generator_optimizer.step()


    #--------------------------------------
    #------  Steps on discriminants -------
    #--------------------------------------


    #---- monet discriminator step ----

    monet_discriminator_optimizer.zero_grad()
    
    real_monet_discr = monet_discriminator(real_monet)
    real_monet_discr_loss = discriminator_real_loss(real_monet_discr)

    fake_monet_discr = monet_discriminator(fake_monet.detach())
    fake_monet_discr_loss = discriminator_fake_loss(fake_monet_discr)

    monet_discr_loss = (real_monet_discr_loss + fake_monet_discr_loss)/2
    monet_discr_loss.backward()
    monet_discriminator_optimizer.step()

    #---- photo discriminator step ----

    photo_discriminator_optimizer.zero_grad()

    real_photo_discr = photo_discriminator(real_photo)
    real_photo_discr_loss = discriminator_real_loss(real_photo_discr)
    
    fake_photo_discr = photo_discriminator(fake_photo.detach())
    fake_photo_discr_loss = discriminator_fake_loss(fake_photo_discr)

    photo_discr_loss = (real_photo_discr_loss + fake_photo_discr_loss)/2
    photo_discr_loss.backward()
    photo_discriminator_optimizer.step()

    #---- saving losses ----
    fake_monet_gen_loss_history.append(fake_monet_discr_loss.item())
    fake_photo_gen_loss_history.append(fake_photo_discr_loss.item())
    cycled_monet_gen_loss_history.append(cycled_monet_gen_loss.item())
    cycled_photo_gen_loss_history.append(cycled_photo_gen_loss.item())
    same_monet_gen_loss_history.append(same_monet_gen_loss.item())
    same_photo_gen_loss_history.append(same_photo_gen_loss.item())
    real_monet_discr_loss_history.append(real_monet_discr_loss.item())
    real_photo_discr_loss_history.append(real_photo_discr_loss.item())

  print('Epoch ' + str(epoch) + '/' + str(n_epoch) + ' done...' )

  monet_generator.eval()
  for photoName in photoNames:
    thisPhotoPath = os.path.join(PHOTO_PATH, photoName + '.jpg')
    photo = cv2.imread(thisPhotoPath)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB) 
    photo = np.transpose(photo,(2,0,1))
    photo = photo/127.5 - 1  
    photo = torch.from_numpy(np.expand_dims(photo,0)).type(torch.FloatTensor).to(device)
    monet = monet_generator(torch.cat((photo,photo),dim=0))
    imgName = 'Photo_To_Monet_' + photoName + '_epoch' + str(epoch) + '.jpg'
    filename = os.path.join(savePath, imgName)
    plt.imsave(filename, np.transpose(np.array(monet[0].detach().cpu()),(1,2,0))*0.5+0.5)
  monet_generator.train()

  photo_generator.eval()
  for monetName in monetNames:
    thisPhotoPath = os.path.join(MONET_PATH, monetName + '.jpg')
    monet = cv2.imread(thisPhotoPath)
    monet = cv2.cvtColor(monet, cv2.COLOR_BGR2RGB) 
    monet = np.transpose(monet,(2,0,1))
    monet = monet/127.5 - 1  
    monet = torch.from_numpy(np.expand_dims(monet,0)).type(torch.FloatTensor).to(device)
    photo = photo_generator(torch.cat((monet,monet),dim=0))
    imgName = 'Monet_To_Photo_' + monetName + '_epoch' + str(epoch) + '.jpg'
    filename = os.path.join(savePath, imgName)
    plt.imsave(filename, np.transpose(np.array(photo[0].detach().cpu()),(1,2,0))*0.5+0.5)
  photo_generator.train()
#      print('Epoch : ',i//len_loader,', progress :', i/len_loader)
#      print('------------- Loss on fake monet with monet discr ', fake_monet_discr_loss.item())
#      print('------------- Loss generator ', gen_loss.item())
#      print('------------- Loss monet discr ', monet_discr_loss.item())
#      print('------------- Loss photo discr ', photo_discr_loss.item())

print('Finished Training')

torch.save(monet_generator.state_dict(), '/gpfs/workdir/houdberta/output_monet/monet_generator_2.save')
torch.save(photo_generator.state_dict(), '/gpfs/workdir/houdberta/output_monet/photo_generator_2.save')
torch.save(monet_discriminator.state_dict(), '/gpfs/workdir/houdberta/output_monet/monet_discriminator_2.save')
torch.save(photo_discriminator.state_dict(), '/gpfs/workdir/houdberta/output_monet/photo_discriminator_2.save')


loss_history_csv = pd.DataFrame(list(zip(fake_monet_gen_loss_history,fake_photo_gen_loss_history,cycled_monet_gen_loss_history,cycled_photo_gen_loss_history,same_monet_gen_loss_history,same_photo_gen_loss_history,real_monet_discr_loss_history,real_photo_discr_loss_history)), columns=['fake_monet_gen_loss_history','fake_photo_gen_loss_history','cycled_monet_gen_loss_history','cycled_photo_gen_loss_history','same_monet_gen_loss_history','same_photo_gen_loss_history','real_monet_discr_loss_history','real_photo_discr_loss_history'])
loss_history_csv.to_csv('/gpfs/workdir/houdberta/output_monet/loss_history.csv')


