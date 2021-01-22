import cv2
import numpy as np
import os
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


# Data Handling Parameters
complete_threshold = 0.05
complete_rate = 0.66

core_threshold = 0.05
core_rate = 0.66


class TrainSet(Dataset):
    def __init__(self, img_root, label_root, data):
        self.data = data

        # Data Handling Parameters
        self.complete_threshold = 0.05
        self.complete_rate = 0.66

        self.core_threshold = 0.05
        self.core_rate = 0.66


        # Images directory
        self.img_root = img_root

        # Images files categorized by ratio
        self.img_path1 = []  # tumor ratio > 5%
        self.img_path2 = []  # tumor ratio < 5%
        self.img_path = []

        # Labels directory
        self.label_root = label_root

        # Labels files categorized by ratio
        self.label_path = []

        self.img_transform = transforms.ColorJitter(brightness=0.2)
        ''' self.transform = Compose([
                    RandomVerticalFlip(),
                    RandomHorizontalFlip(),
                    RandomAffine(degrees=(-20,20),translate=(0.1,0.1),
                                 scale=(0.9,1.1), shear=(-0.2,0.2)),
                    ElasticTransform(alpha=720, sigma=24)])
        self.totensor = transforms.ToTensor() '''

        if self.data == 'complete':
            self.patch_threshold = self.complete_threshold
            self.data_rate = self.complete_rate
        elif self.data == 'core':
            self.patch_threshold = self.core_threshold
            self.data_rate = self.core_rate

        # loading image path from disk
        ''' img_path = []
        for folder_name in os.listdir(self.img_root):
            img_path += glob.glob(os.path.join(self.img_root, folder_name, "*.jpg"))

        #loading label path from disk
        label_path = []
        for folder_name in os.listdir(self.label_root):
            label_path += glob.glob(os.path.join(self.label_root, folder_name, "*.jpg")) '''

        dict_img_path = {}
        for folder_name in os.listdir(img_root):
            IRM_ID = int(folder_name.split('_')[-2]) - 1
            dict_img_path[IRM_ID] = {}
            files = glob.glob(os.path.join(img_root, folder_name, "*.jpg"))
            for filename in files:
                image_ID = int(filename.split('_')[-1].split('.')[0])
                dict_img_path[IRM_ID][image_ID] = filename

        # loading label path from disk
        dict_label_path = {}
        for folder_name in os.listdir(label_root):
            IRM_ID = int(folder_name.split('_')[-2]) - 1
            dict_label_path[IRM_ID] = {}
            files = glob.glob(os.path.join(label_root, folder_name, "*.jpg"))
            for filename in files:
                image_ID = int(filename.split('_')[-1].split('.')[0])
                dict_label_path[IRM_ID][image_ID] = filename

        img_path = []
        label_path = []
        for i in dict_img_path.keys():
            for j in dict_img_path[i].keys():
                img_path.append(dict_img_path[i][j])
                label_path.append(dict_label_path[i][j])

        # Store all image path
        self.img_path = img_path
        self.label_path = label_path

        # Store Dict img path
        self.dict_img_path = dict_img_path
        self.dict_label_path = dict_label_path

        # loading all labels
        for i in dict_label_path.keys():
            for j in range(155):
                label = cv2.imread(dict_label_path[i][j], cv2.IMREAD_GRAYSCALE)

                if self.data == 'complete':
                    tumor_ratio = (label > 25).astype(np.uint8).sum()
                elif self.data == 'core':
                    l1 = (label > 25).astype(np.uint8)
                    l2 = (label < 75).astype(np.uint8)
                    label1 = np.logical_and(l1, l2).astype(np.uint8)
                    l1 = (label > 175).astype(np.uint8)
                    l2 = (label < 225).astype(np.uint8)
                    label2 = np.logical_and(l1, l2).astype(np.uint8)
                    tumor_ratio = (np.logical_or(
                        label1, label2).astype(np.uint8)).sum()
                    del label1, label2
                else:
                    l1 = (label > 175).astype(np.uint8)
                    l2 = (label < 225).astype(np.uint8)
                    tumor_ratio = (np.logical_and(
                        l1, l2).astype(np.uint8)).sum()
                tumor_ratio /= label.shape[0]*label.shape[1]

                # Put path in list 2 if tumor ratio > threshold (else put in list 1)
                if tumor_ratio > self.patch_threshold:  # CHANGE FOR ENHANCING
                    self.img_path2.append([i, j])
                else:
                    self.img_path1.append([i, j])

    def __len__(self):
        # TODO: Change len (ex. len(img_path2)*1.5)
        return len(self.img_path)

    def __getitem__(self, idx):
        if np.random.choice(2, 1, p=[1-self.data_rate, self.data_rate]) == 0:
            idx = idx % len(self.img_path1)
            brainID = self.img_path1[idx][0]
            layerID = self.img_path1[idx][1]
        else:
            idx = np.random.randint(len(self.img_path2))
            brainID = self.img_path2[idx][0]
            layerID = self.img_path2[idx][1]

        # Il faut gérer les bords (quand idx = 0 ou 155)
        # brainID = idx // 155
        # layerID = idx % 155

        if layerID == 0 or layerID == 154:
            layerID = np.random.randint(1, 154)

        img1_path = self.dict_img_path[brainID][layerID - 1]
        img2_path = self.dict_img_path[brainID][layerID]
        img3_path = self.dict_img_path[brainID][layerID + 1]
        label_path = self.dict_label_path[brainID][layerID]

        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = Image.fromarray(img1)

        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = Image.fromarray(img2)

        img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
        img3 = Image.fromarray(img3)

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Select tumor aspect
        if self.data == 'complete':
            label[label < 25] = 0
            label[label >= 25] = 1
            label = label.astype(np.uint8)
        elif self.data == 'core':
            # A modifier (150 -> 200)
            l1 = (label > 25).astype(np.uint8)
            l2 = (label < 75).astype(np.uint8)
            label1 = np.logical_and(l1, l2).astype(np.uint8)
            l1 = (label > 175).astype(np.uint8)
            l2 = (label < 225).astype(np.uint8)
            label2 = np.logical_and(l1, l2).astype(np.uint8)
            label = np.logical_or(label1, label2).astype(np.uint8)
            del label1, label2
        else:
            l1 = (label > 175).astype(np.uint8)
            l2 = (label < 225).astype(np.uint8)
            label = np.logical_and(l1, l2).astype(np.uint8)

        label = Image.fromarray(label)
        # img, label = self.transform(img,label)
        # img = self.img_transform(img)

        label = np.expand_dims(np.array(label), 0)
        # label = np.expand_dims(label, 0)

        img1 = np.expand_dims(np.array(img1), 0)
        img1 = img1.astype(np.float32)
        img1 = (img1/127.5) - 1

        img2 = np.expand_dims(np.array(img2), 0)
        img2 = img2.astype(np.float32)
        img2 = (img2/127.5) - 1

        img3 = np.expand_dims(np.array(img3), 0)
        img3 = img3.astype(np.float32)
        img3 = (img3/127.5) - 1

        img = np.empty((3, 240, 240))
        img[0] = img1
        img[1] = img2
        img[2] = img3
        img = img.astype(np.float32)

        label = label.astype('int')
        # label = np.concatenate((np.absolute(label-1) , label),axis=0)
        label = torch.from_numpy(label)
        return torch.from_numpy(img), label.type(torch.LongTensor), img2_path


class TestSet(Dataset):
    def __init__(self, img_root, label_root, data='core'):
        self.data = data
        # Images directory
        self.img_root = img_root

        # Labels directory
        self.label_root = label_root

        self.img_path = []
        self.label_path = []

        dict_img_path = {}
        for folder_name in os.listdir(img_root):
            IRM_ID = int(folder_name.split('_')[-2]) - 1
            dict_img_path[IRM_ID] = {}
            files = glob.glob(os.path.join(img_root, folder_name, "*.jpg"))
            for filename in files:
                image_ID = int(filename.split('_')[-1].split('.')[0])
                dict_img_path[IRM_ID][image_ID] = filename

        # loading label path from disk
        dict_label_path = {}
        for folder_name in os.listdir(label_root):
            IRM_ID = int(folder_name.split('_')[-2]) - 1
            dict_label_path[IRM_ID] = {}
            files = glob.glob(os.path.join(label_root, folder_name, "*.jpg"))
            for filename in files:
                image_ID = int(filename.split('_')[-1].split('.')[0])
                dict_label_path[IRM_ID][image_ID] = filename

        img_path = []
        label_path = []
        MRI_id = []
        for i in dict_img_path.keys():
            for j in dict_img_path[i].keys():
                if j == 0 or j == 154:
                    continue
                img_path.append(dict_img_path[i][j])
                label_path.append(dict_label_path[i][j])
                MRI_id.append([i, j])

        # Store all image path
        self.img_path = img_path
        self.label_path = label_path
        self.MRI_id = MRI_id

        # Store Dict img path
        self.dict_img_path = dict_img_path
        self.dict_label_path = dict_label_path


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
      
        brainID = self.MRI_id[idx][0]
        layerID = self.MRI_id[idx][1]

        img1_path = self.dict_img_path[brainID][layerID - 1]
        img2_path = self.dict_img_path[brainID][layerID]
        img3_path = self.dict_img_path[brainID][layerID + 1]
        label_path = self.dict_label_path[brainID][layerID]

        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = Image.fromarray(img1)

        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = Image.fromarray(img2)

        img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
        img3 = Image.fromarray(img3)

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Select tumor aspect
        if self.data == 'complete':
            label[label < 25] = 0
            label[label >= 25] = 1
            label = label.astype(np.uint8)
        elif self.data == 'core':
            # A modifier (150 -> 200)
            l1 = (label > 25).astype(np.uint8)
            l2 = (label < 75).astype(np.uint8)
            label1 = np.logical_and(l1, l2).astype(np.uint8)
            l1 = (label > 175).astype(np.uint8)
            l2 = (label < 225).astype(np.uint8)
            label2 = np.logical_and(l1, l2).astype(np.uint8)
            label = np.logical_or(label1, label2).astype(np.uint8)
            del label1, label2
        else:
            l1 = (label > 175).astype(np.uint8)
            l2 = (label < 225).astype(np.uint8)
            label = np.logical_and(l1, l2).astype(np.uint8)

        label = Image.fromarray(label)
        label = np.expand_dims(np.array(label), 0)

        img1 = np.expand_dims(np.array(img1), 0)
        img1 = img1.astype(np.float32)
        img1 = (img1/127.5) - 1

        img2 = np.expand_dims(np.array(img2), 0)
        img2 = img2.astype(np.float32)
        img2 = (img2/127.5) - 1

        img3 = np.expand_dims(np.array(img3), 0)
        img3 = img3.astype(np.float32)
        img3 = (img3/127.5) - 1

        img = np.empty((3, 240, 240))
        img[0] = img1
        img[1] = img2
        img[2] = img3
        img = img.astype(np.float32)

        label = label.astype('int')
        label = torch.from_numpy(label)
        return torch.from_numpy(img), label.type(torch.LongTensor), img2_path

def Load_Test_MRI(self, img_root, data='complete'):
    self.data = data

    # Images directory
    self.img_root = img_root

    # Images files categorized by ratio
    self.img_path = []

    # Labels directory
    self.label_root = label_root

    # Labels files categorized by ratio
    self.label_path = []

    dict_img_path = {}
    for folder_name in os.listdir(img_root):
        IRM_ID = int(folder_name.split('_')[-2]) - 1
        dict_img_path[IRM_ID] = {}
        files = glob.glob(os.path.join(img_root, folder_name, "*.jpg"))
        for filename in files:
            image_ID = int(filename.split('_')[-1].split('.')[0])
            dict_img_path[IRM_ID][image_ID] = filename

    img_path = []
    MRI_id = []
    for i in dict_img_path.keys():
        for j in dict_img_path[i].keys():
            img_path.append(dict_img_path[i][j])
            MRI_id.append([i, j])

    # Store all image path
    self.img_path = img_path

    self.MRI_id = MRI_id

    # Store Dict img path
    self.dict_img_path = dict_img_path


def __len__(self):
    return len(self.dict_img_path.keys())


def __getitem__(self, idx):

    brainID = self.MRI_id[idx][0]
    layerID = self.MRI_id[idx][1]

    # Il faut gérer les bords (quand idx = 0 ou 155)
    # brainID = idx // 155
    # layerID = idx % 155

    if layerID == 0 or layerID == 154:
        layerID = np.random.randint(1, 154)

    img1_path = self.dict_img_path[brainID][layerID - 1]
    img2_path = self.dict_img_path[brainID][layerID]
    img3_path = self.dict_img_path[brainID][layerID + 1]
    label_path = self.dict_label_path[brainID][layerID]

    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1 = Image.fromarray(img1)

    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img2 = Image.fromarray(img2)

    img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
    img3 = Image.fromarray(img3)

    # Load label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Select tumor aspect
    if self.data == 'complete':
        label[label < 25] = 0
        label[label >= 25] = 1
        label = label.astype(np.uint8)
    elif self.data == 'core':
        # A modifier (150 -> 200)
        l1 = (label > 25).astype(np.uint8)
        l2 = (label < 75).astype(np.uint8)
        label1 = np.logical_and(l1, l2).astype(np.uint8)
        l1 = (label > 175).astype(np.uint8)
        l2 = (label < 225).astype(np.uint8)
        label2 = np.logical_and(l1, l2).astype(np.uint8)
        label = np.logical_or(label1, label2).astype(np.uint8)
        del label1, label2
    else:
        l1 = (label > 175).astype(np.uint8)
        l2 = (label < 225).astype(np.uint8)
        label = np.logical_and(l1, l2).astype(np.uint8)

    label = Image.fromarray(label)
    # img, label = self.transform(img,label)
    # img = self.img_transform(img)

    label = np.expand_dims(np.array(label), 0)
    # label = np.expand_dims(label, 0)

    img1 = np.expand_dims(np.array(img1), 0)
    img1 = img1.astype(np.float32)
    img1 = (img1/127.5) - 1

    img2 = np.expand_dims(np.array(img2), 0)
    img2 = img2.astype(np.float32)
    img2 = (img2/127.5) - 1

    img3 = np.expand_dims(np.array(img3), 0)
    img3 = img3.astype(np.float32)
    img3 = (img3/127.5) - 1

    img = np.empty((3, 240, 240))
    img[0] = img1
    img[1] = img2
    img[2] = img3
    img = img.astype(np.float32)

    label = label.astype('int')
    #label = np.concatenate((np.absolute(label-1) , label),axis=0)
    label = torch.from_numpy(label)
    return torch.from_numpy(img), label.type(torch.LongTensor), img2_path
