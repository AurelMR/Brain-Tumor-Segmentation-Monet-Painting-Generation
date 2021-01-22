import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision.transforms import RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, ColorJitter, Compose, RandomPerspective, Resize, RandomCrop
from PIL import Image
import numpy.random as rd

def image_rotate_random(image_pil):
  p_rotate90 = rd.rand()

  if p_rotate90>0.75:
    image = image_pil.rotate(270)
  elif p_rotate90>0.5:
    image = image_pil.rotate(180)
  elif p_rotate90>0.25:
    image = image_pil.rotate(90)
  else:
    image = image_pil

  p_rotate15 = rd.random()
  if p_rotate15 >0.5:
    angle = 15*(p_rotate15-0.75)*4
    resize = 5 + 128*np.sin(np.deg2rad(np.abs(angle)))*np.sin(np.deg2rad(90-np.abs(angle)))/np.sin(np.deg2rad(45-np.abs(angle)))
    image = image_pil.rotate(angle,expand=True)
    image = image.crop((resize,resize,image.size[0]-resize,image.size[0]-resize))
    image = image.resize((256,256))
  return image

def image_tensor_transform(image_tensor):
  transform_vect = [ RandomVerticalFlip(), RandomHorizontalFlip() ]
  
  p_resize = rd.rand()
  p_rotate = rd.rand()

  if p_resize > 0.5: 
    size = int(256 + 2*(p_resize-0.5) * 50)
    transform_vect.append(Resize((size, size)))
    transform_vect.append(RandomCrop((256,256)))
   
  transform = Compose(transform_vect)
  return transform(image_tensor)

class ImageLoader(Dataset):
  def __init__(self,monet_path,photo_path):
    self.monet_img_paths = glob.glob(os.path.join(monet_path,'*.jpg'))
    self.photo_img_paths = glob.glob(os.path.join(photo_path,'*.jpg'))

  def __len__(self):
    return max(len(self.monet_img_paths),len(self.photo_img_paths))
  
  def __getitem__(self,idx):
    monet_images_path = self.monet_img_paths[idx % len(self.monet_img_paths)]
    photo_images_path = self.photo_img_paths[rd.randint(0,len(self.photo_img_paths))]

    monet = cv2.imread(monet_images_path)
    monet = cv2.cvtColor(monet, cv2.COLOR_BGR2RGB) 
    monet = Image.fromarray(monet)
    monet = image_rotate_random(monet)
    monet = np.array(monet)
    monet = np.transpose(monet,(2,0,1))
    monet = monet/127.5 - 1  
    monet = image_tensor_transform(torch.from_numpy(monet).type(torch.FloatTensor))
    
    photo = cv2.imread(photo_images_path)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    photo = Image.fromarray(photo)
    photo = image_rotate_random(photo)
    photo = np.array(photo) 
    photo = np.transpose(photo,(2,0,1))
    photo = photo/127.5 - 1  
    photo = image_tensor_transform(torch.from_numpy(photo).type(torch.FloatTensor))

    return monet,photo,monet_images_path,photo_images_path

    