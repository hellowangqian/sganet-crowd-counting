# -*- coding: utf-8 -*-
"""
==========================
**Author**: Qian Wang, qian.wang173@hotmail.com
"""


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
plt.ion()   # interactive mode
from model import CANNet
from model_mcnn import MCNN
from model_cffnet import CFFNet
from model_csrnet import CSRNet
from model_sanet import SANet
from model_tednet import TEDNet
from myInception_segLoss import headCount_inceptionv3
from generate_density_map import generate_multi_density_map,generate_density_map

IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    """
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
    """
    d = os.path.join(dir,'images')
    for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    image_path = os.path.join(root, fname)
                    head,tail = os.path.split(root)
                    label_path = os.path.join(head,'ground_truth','GT_'+fname[:-4]+'.mat')
                    item = [image_path, label_path]
                    images.append(item)

    return images

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train',extensions=IMG_EXTENSIONS,patch_size=128,num_patches_per_image=4):
        self.samples = make_dataset(data_dir,extensions)
        self.image_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.patch_size = patch_size
        self.numPatches = num_patches_per_image
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        img_file,label_file = self.samples[idx]
        image = cv2.imread(img_file)
        height, width, channel = image.shape
        annPoints = scipy.io.loadmat(label_file)
        annPoints = annPoints['image_info'][0][0][0][0][0]
        positions = generate_density_map(shape=image.shape,points=annPoints,f_sz=15,sigma=4)
        fbs = generate_density_map(shape=image.shape,points=annPoints,f_sz=25,sigma=1)
        fbs = np.int32(fbs>0)
        targetSize = [self.patch_size,self.patch_size]
        height, width, channel = image.shape
        if height < targetSize[0] or width < targetSize[1]:
            image = cv2.resize(image,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count = positions.sum()
            max_value = positions.max()
            # down density map
            positions = cv2.resize(positions, (np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count2 = positions.sum()
            positions = np.minimum(positions*count/(count2+1e-8),max_value*10)
            fbs = cv2.resize(fbs,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            fbs = np.int32(fbs>0)
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        # transpose from h x w x channel to channel x h x w
        image = image.transpose(2,0,1)
        numPatches = self.numPatches
        if self.phase == 'train':
            patchSet, countSet, fbsSet = getRandomPatchesFromImage(image,positions,fbs,targetSize,numPatches)
            x = np.zeros((patchSet.shape[0],3,targetSize[0],targetSize[1]))
            if self.transform:
              for i in range(patchSet.shape[0]):
                #transpose to original:h x w x channel
                x[i,:,:,:] = self.transform(np.uint8(patchSet[i,:,:,:]).transpose(1,2,0))
            patchSet = x
        if self.phase == 'val' or self.phase == 'test':
            patchSet, countSet, fbsSet = getAllFromImage(image, positions, fbs)
            patchSet[0,:,:,:] = self.transform(np.uint8(patchSet[0,:,:,:]).transpose(1,2,0))
        return patchSet, countSet, fbsSet

def getRandomPatchesFromImage(image,positions,fbs,target_size,numPatches):
    # generate random cropped patches with pre-defined size, e.g., 224x224
    imageShape = image.shape
    if np.random.random()>0.5:
        for channel in range(3):
            image[channel,:,:] = np.fliplr(image[channel,:,:])
        positions = np.fliplr(positions)
        fbs = np.fliplr(fbs)
    patchSet = np.zeros((numPatches,3,target_size[0],target_size[1]))
    # generate density map
    countSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    fbsSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1]-target_size[0]+1)#x-height
        topLeftY = np.random.randint(imageShape[2]-target_size[1]+1)#y-width
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        patchSet[i,:,:,:] = thisPatch
        # density map
        position = positions[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        fb = fbs[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        position = position.reshape((1, position.shape[0], position.shape[1]))
        fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
        countSet[i,:,:,:] = position
        fbsSet[i,:,:,:] = fb
    return patchSet, countSet, fbsSet

def getAllPatchesFromImage(image,positions,target_size):
    # generate all patches from an image for prediction
    nchannel,height,width = image.shape
    nRow = np.int(height/target_size[1])
    nCol = np.int(width/target_size[0])
    target_size[1] = np.int(height/nRow)
    target_size[0] = np.int(width/nCol)
    patchSet = np.zeros((nRow*nCol,3,target_size[1],target_size[0]))
    for i in range(nRow):
      for j in range(nCol):
        patchSet[i*nCol+j,:,:,:] = image[:,i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
    return patchSet

def getAllFromImage(image,positions,fbs):
    nchannel, height, width = image.shape
    patchSet =np.zeros((1,3,height, width))
    patchSet[0,:,:,:] = image[:,:,:]
    countSet = positions.reshape((1,1,positions.shape[0], positions.shape[1]))
    fbsSet = fbs.reshape((1,1,fbs.shape[0], fbs.shape[1]))
    return patchSet, countSet, fbsSet

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def train_model(model, optimizer, scheduler, num_epochs=100, seg_loss=False, cl_loss=False, test_step=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae_val = 1e6
    best_mae_by_val = 1e6
    best_mae_by_test = 1e6
    best_mse_by_val = 1e6
    best_mse_by_test = 1e6
    criterion1 = nn.MSELoss(reduce=False) # for density map loss
    criterion2 = nn.BCELoss() # for segmentation map loss
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0        
        # Iterate over data.
        for index, (inputs, labels, fbs) in enumerate(dataloaders['train']):
            labels = labels*100
            labels = skimage.measure.block_reduce(labels.cpu().numpy(),(1,1,1,4,4),np.sum)
            fbs = skimage.measure.block_reduce(fbs.cpu().numpy(),(1,1,1,4,4),np.max)
            fbs = np.float32(fbs>0)
            labels = torch.from_numpy(labels)
            fbs = torch.from_numpy(fbs)
            labels = labels.to(device)
            fbs = fbs.to(device)
            inputs = inputs.to(device)
            inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
            labels = labels.view(-1,labels.shape[3],labels.shape[4])
            fbs = fbs.view(-1,fbs.shape[3],fbs.shape[4])
            inputs = inputs.float()
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                output,fbs_out = model(inputs)
                loss_den = criterion1(output, labels)
                loss_seg = criterion2(fbs_out, fbs)
                if cl_loss:
                    th = 0.1*epoch+5 #cl2
                else:
                    th=1000 # no curriculum loss when th is set a big number
                weights = th/(F.relu(labels-th)+th)
                loss_den = loss_den*weights
                loss_den = loss_den.sum()/weights.sum()
                if seg_loss:
                    loss = loss_den + 20*loss_seg
                else:
                    loss = loss_den

                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
               
        scheduler.step()    
        epoch_loss = running_loss / dataset_sizes['train']            
        
        print('Train Loss: {:.4f}'.format(epoch_loss))
        print()
        if epoch%test_step==0:
            tmp,epoch_mae,epoch_mse,epoch_mre=test_model(model,optimizer,'val')
            tmp,epoch_mae_test,epoch_mse_test,epoch_mre_test = test_model(model,optimizer,'test')
            if  epoch_mae < best_mae_val:
                best_mae_val = epoch_mae
                best_mae_by_val = epoch_mae_test
                best_mse_by_val = epoch_mse_test
                best_epoch_val = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch_mae_test < best_mae_by_test:
                best_mae_by_test = epoch_mae_test
                best_mse_by_test = epoch_mse_test
                best_epoch_test = epoch
            print()
            print('best MAE and MSE by val:  {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_val,best_mse_by_val, best_epoch_val))
            print('best MAE and MSE by test: {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_test,best_mse_by_test, best_epoch_test))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    # Iterate over data.
    for index, (inputs, labels, fbs) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,labels.shape[3],labels.shape[4])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(False):
            outputs,fbs_out = model(inputs)
            outputs = outputs.to(torch.device("cpu")).numpy()/100
            pred_count = outputs.sum()
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        # backward + optimize only if in training phase
        mse = mse + np.square(pred_count-true_count)
        mae = mae + np.abs(pred_count-true_count)
        mre = mre + np.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = np.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(f'MAE:{mae:2.2f}, RMSE:{mse:2.2f}, MRE:{mre:2.4f}')
    time_elapsed = time.time() - since
    return pred,mae,mse,mre

#####################################################################
# set parameters here
seg_loss = True
cl_loss = True
test_step = 1
batch_size = 6
num_workers = 4
patch_size = 128
num_patches_per_image = 4
data_dir = './data/part_B_final/'

# define data set
image_datasets = {x: ShanghaiTechDataset(data_dir+x+'_data', 
                        phase=x, 
                        transform=data_transforms[x],
                        patch_size=patch_size,
                        num_patches_per_image=num_patches_per_image)
                    for x in ['train','test']}
image_datasets['val'] = ShanghaiTechDataset(data_dir+'train_data',
                            phase='val',
                            transform=data_transforms['val'],
                            patch_size=patch_size,
                            num_patches_per_image=num_patches_per_image)
## split the data into train/validation/test subsets
indices = list(range(len(image_datasets['train'])))
split = np.int(len(image_datasets['train'])*0.2)

val_idx = np.random.choice(indices, size=split, replace=False)
train_idx = indices#list(set(indices)-set(val_idx))
test_idx = range(len(image_datasets['test']))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetSampler(test_idx)

train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'],batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'],batch_size=1,sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'],batch_size=1,sampler=test_sampler, num_workers=num_workers)

dataset_sizes = {'train':len(train_idx),'val':len(val_idx),'test':len(image_datasets['test'])}
dataloaders = {'train':train_loader,'val':val_loader,'test':test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# define models and training
model = headCount_inceptionv3(pretrained=True)
# model = MCNN()
# model = SANet()
# model = TEDNet(use_bn=True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

model = train_model(model, optimizer, exp_lr_scheduler,
                    num_epochs=501, 
                    seg_loss=seg_loss, 
                    cl_loss=cl_loss, 
                    test_step=test_step)
                    
pred,mae,mse,mre = test_model(model,optimizer,'test')
scipy.io.savemat('./results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
model_dir = './'
torch.save(model.state_dict(), model_dir+'saved_model.pt')

