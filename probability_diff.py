"""
Created on Tue Aug 20 10:05:11 2019

@author: Ankita
"""

from __future__ import print_function, division
import torch
import os
import copy
import time
import csv
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

torch.manual_seed(0)

## Calculating impact score
def impact(p1, p2, w1, w2):
    x = 0
    for i in range(len(p1)):
        for j in range(len(p1[i])):
            x += abs((p2[i][j][0] - p1[i][j][0]) / (w2 - w1))
    return x


## Preprocessing
class MyDataset(Dataset):
    
  def __init__(self, img, Transform = None, mode='train'): 
    super(MyDataset, self).__init__()
    self.images = img
    self.transform = Transform 
    self.mode = mode 
    
  def __len__(self): 
    return self.images.shape[0] 

  def __getitem__(self, index): 
    img = self.images[index] 
    if (self.transform != None): 
      x = self.transform(img)
      x = Variable(x).cuda()
    else: 
      x = img
      x = Variable(x).cuda()
    return x

class ToTensor(object):

    def __call__(self, x):
        image = x
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        return image

        
    
## Using GPU
use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## model loading
since = time.time()
nclass = 7
vgg = models.vgg16_bn(pretrained=False)
for param in vgg.features.parameters():
    param.requires_grad = False
classifier, features = list(vgg.classifier)[:-8], (vgg.features)[:-20]
classifier.extend(nn.Sequential(nn.Linear(12544, nclass),nn.BatchNorm1d(nclass),nn.ReLU(inplace=True)))
vgg.classifier, vgg.features = nn.Sequential(*classifier), nn.Sequential(*features)
vgg = vgg.eval()

## load data
if use_gpu:
    vgg.cuda()
best_model_wts = copy.deepcopy(vgg.state_dict())
os.chdir('/home/ankitaC/Ankita/164209861/vgg/weights')
vgg.load_state_dict(torch.load('chkpt_epoch_5.pt'))
os.chdir('/home/ankitaC/Ankita/reduction')
data = np.load('patch_test.npy')
test_data = data
print(test_data.shape)
batch_size = 10000
transformers = transforms.Compose ([ToTensor()])
data_test = MyDataset(test_data,transformers)
testloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)


## testing
p1 = list()
vgg.train(False)
for img in testloader:
    img = img.view([-1,3,16,16])
    img = img.to(device)
    outputs = vgg(img)
    p = (outputs.data.cpu().numpy())
    p1.append(p)
p1 = np.array(p1)
#print((p1[0][0][0]))

## changing filter weights for every layer
layers = np.array([0, 3, 7, 10, 14, 17, 20])
for i in (range(len(layers))):
    probability = list()
    features = list(vgg.features)
    filtr = features[layers[i]].weight.data
    vgg.features = nn.Sequential(*features)
    for j in tqdm(range(filtr.shape[0])):
        os.chdir('/home/ankitaC/Ankita/164209861/vgg/weights')
        vgg.load_state_dict(torch.load('chkpt_epoch_5.pt'))
        os.chdir('/home/ankitaC/Ankita/reduction')
        features = list(vgg.features)
        filtr = features[layers[i]].weight.data
        x = filtr[j].cpu().numpy()
        w1 = np.sum(np.sum(x, axis = 0))
        alter = filtr[j] + 1
        x = alter.cpu().numpy()
        w2 = np.sum(np.sum(x, axis = 0))
        features[layers[i]].weight.data[j] = alter
        vgg.features = nn.Sequential(*features)
        p2 = list()
        for img in testloader:
            img = img.view([-1,3,16,16])
            img = img.to(device)
            vgg.train(False)
            outputs = vgg(img)
            p = (outputs.data.cpu().numpy())
            p2.append(p)
        p2 = np.array(p2)
        #print(p2[0,0,0])
        probability.append(impact(p1, p2, w1, w2))
    probability = np.array(probability)
    print(np.argmax(probability))
    csvfile = 'output_label_num' + str(i) + '.csv'
    with open(csvfile, "a") as output1:
        writer = csv.writer(output1, lineterminator='\n')
        writer.writerow(probability)
