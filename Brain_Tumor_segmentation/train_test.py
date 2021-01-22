import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import cv2

from dataset import TrainSet, TestSet
from UNet import UNet
from tqdm import tqdm
import os
import pandas as pd

# IMG_ROOT = 'C://Users//ahoud//Documents//Cours ECP//3A//IGR//Scripts//dataset//train_image'
# LABEL_ROOT = 'C://Users//ahoud//Documents//Cours ECP//3A//IGR//Scripts//dataset//train_label'

IMG_ROOT = '/gpfs/workdir/houdberta/dataset/train_image'
LABEL_ROOT = '/gpfs/workdir/houdberta/dataset/train_label'


BATCH_SIZE = 16
data_type = 'core'
# Load dataset
print('Loading TrainingSet') # --------------------------------------------------------
trainset = TrainSet(IMG_ROOT, LABEL_ROOT, data_type)
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True)
print('Done') # -----------------------------------------------------------------------

# Defining model and optimization methode
device = 'cuda:0'
#device = 'cpu'
unet = UNet(in_channel=3, class_num=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005, amsgrad=True)

epochs = 15
lsize = len(trainloader)
itr = 0
p_itr = 100  # print every N iteration
unet.train()
tloss = 0
loss_history = []

print('Starting Training') # --------------------------------------------------------
for epoch in range(epochs):
    print('Epoch: [', epoch, ':', epochs, ']')
    #with tqdm(total=lsize) as pbar:
    for x, y, path in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = unet(x)
        loss = criterion(output, y[:, 0, :, :].to(device))
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        loss_history.append(loss.item())

        if itr % p_itr == 0:
            print('TrainLoss: ', tloss)
            tloss = 0
        """ if itr % p_itr == 0:
            pbar.update(10)
            pbar.set_description(
                "Train Loss: {:.4f}".format(tloss/p_itr))
            tloss = 0 """
        itr += 1
        print('done ...')

loss_history_csv = pd.DataFrame(loss_history, columns=['Loss'])
loss_history_csv.to_csv('/gpfs/workdir/houdberta/output/loss_history.csv')


print('Finished Training') # -------------------------------------------------------
torch.save(unet.state_dict(), '/gpfs/workdir/houdberta/output/network.save')


test_img_root = "/gpfs/workdir/houdberta/dataset/test_image"
test_label_root = "/gpfs/workdir/houdberta/dataset/test_label"

BATCH_SIZE = 16
testset = TestSet(test_img_root, test_label_root, data_type)
testloader = DataLoader(testset, BATCH_SIZE, shuffle=True)

output_root = '/gpfs/workdir/houdberta/output_pred'


print('Starting testing') # --------------------------------------------------------
unet.eval()
for x, y, path in testloader:
    x, y = x.to(device), y.to(device)
    output = unet(x)
    output_soft = softmax(output, dim=1)
    output_array = np.array(output_soft.cpu().detach())
    label_pred = np.argmax(output_array,axis=1)
    for i in range(label_pred.shape[0]):
        mask_pred = label_pred[i]
        current_path = path[i]
        stepLayer = current_path.split('/')[-1].split('_')
        brainID = stepLayer[-3]
        layerID = stepLayer[-1].split('.')[0]

        img_dir = 'BraTS20_TestPred_' + brainID
        output_path = os.path.join(output_root, img_dir)
        try:
            os.mkdir(output_path)
        except:
            pass
        #print(brainID)
        #print(layerID)
        #print(mask_pred.shape)
        filename = 'test_' + brainID + '_' + layerID + '_.jpg'
        filename = os.path.join(output_path, filename)
        cv2.imwrite(filename, mask_pred)

print('Finished testing') # --------------------------------------------------------

