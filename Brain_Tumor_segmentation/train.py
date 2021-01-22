import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from dataset import TrainSet, TestSet
from UNet import UNet
from tqdm import tqdm
import os

# IMG_ROOT = '../Scripts/dataset/train_image/'
# LABEL_ROOT = '../Scripts/dataset/train_label/'

IMG_ROOT = '/gpfs/workdir/houdberta/dataset/train_image'
LABEL_ROOT = '/gpfs/workdir/houdberta/dataset/train_label'

BATCH_SIZE = 8


# Load dataset
trainset = TrainSet(IMG_ROOT, LABEL_ROOT, 'core')
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True)

# Defining model and optimization methode
unet = UNet(in_channel=3, class_num=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005, amsgrad=True)

epochs = 60
lsize = len(trainloader)
itr = 0
p_itr = 10  # print every N iteration
unet.train()
tloss = 0
loss_history = []

for epoch in range(epochs):
    with tqdm(total=lsize) as pbar:
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
                pbar.update(10)
                pbar.set_description(
                    "Train Loss: {:.4f}".format(tloss/p_itr))
                tloss = 0
            itr += 1

            break


print('Finished Training')

torch.save(unet.state_dict(), '/gpfs/workdir/houdberta/output/network.save')


########################################################################

# Testing model

"""
TEST_IMG_ROOT = '$WORKDIR/dataset/test_image'
TEST_LABEL_ROOT = '$WORKDIR/dataset/test_label'
TEST_BATCH_SIZE = 8

testset = TrainSet(TEST_IMG_ROOT, TEST_LABEL_ROOT, 'core')
testloader = DataLoader(testset, BATCH_SIZE, shuffle=True)

unet.eval()

dict_img_path = {}
for folder_name in os.listdir(TEST_IMG_ROOT):
    IRM_ID = int(folder_name.split('_')[-2]) - 1
    dict_img_path[IRM_ID] = {}
    files = glob.glob(os.path.join(TEST_IMG_ROOT, folder_name, "*.jpg"))
    for filename in files:
        image_ID = int(filename.split('_')[-1].split('.')[0])
        dict_img_path[IRM_ID][image_ID] = filename

dict_id = list(dict_img_path)
dict_id.sort()

for brainID in dict_id:
    for layerID in range(1,154):
        img1_path = dict_img_path[brainID][layerID - 1]
        img2_path = dict_img_path[brainID][layerID]
        img3_path = dict_img_path[brainID][layerID + 1]

        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = Image.fromarray(img1)

        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = Image.fromarray(img2)

        img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
        img3 = Image.fromarray(img3)

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

        x = torch.from_numpy(img)

        x = x.to(device)
        output = unet(x)
        output_soft = softmax(output, dim=1)
        output_array = np.array(output_soft.to('cpu').detach())
        label_pred = np.argmax(output_array, axis=1)

        label_pred = label_pred * 50
        label_pred = label_pred.astype(np.uint8)

        try:
            os.mkdir(output_root)
        except:
            pass
        try:
            os.mkdir(output_path)
        except:
            pass

        output_path = '$WORKDIR/output_pred/BraTS20_' + str(brainID) + '_PredictedLabel'

        filename = os.path.join(output_path, 'BraTS20_' str(brainID) + '_PredictedLabel_' + str(layerID) + '.jpg')
        cv2.imwrite(filename, label_pred)
"""

#################################################################################

""" 
def train(epochs, IMG_ROOT, LABEL_ROOT, BATCH_SIZE=8, lr=0.002, device='cpu'):

    # Load dataset
    trainset = TrainSet(IMG_ROOT, LABEL_ROOT, 'core')
    loader = DataLoader(trainset, BATCH_SIZE, shuffle=True)

    # Defining model and optimization methode
    unet = UNet(in_channel=3, class_num=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, amsgrad=True)

    lsize = len(loader)
    itr = 0
    p_itr = 10  # print every N iteration
    unet.train()
    tloss = 0
    loss_history = []

    for epoch in range(epochs):
        with tqdm(total=lsize) as pbar:
            for x, y, path in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = unet(x)
                loss = criterion(output, y[:, 0, :, :].to(device))
                loss.backward()
                optimizer.step()
                tloss += loss.item()
                loss_history.append(loss.item())

                if itr % p_itr == 0:
                    pbar.update(10)
                    pbar.set_description(
                        "Train Loss: {:.4f}".format(tloss/p_itr))
                    tloss = 0
                itr += 1

                break

    return loss_history
 
 train(1, IMG_ROOT, LABEL_ROOT, BATCH_SIZE=8, lr=0.002, device='cpu')
 
 """
