from sam import SAMSGD
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
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
#import Image
from PIL import Image
import sys
import timm

if len(sys.argv) >= 2:
    BATCH_SIZE = int(sys.argv[1])

TRAIN_RATIO =1.0
if len(sys.argv) == 3:
    TRAIN_RATIO = float(sys.argv[2])
print("Batch size: ", BATCH_SIZE)

typelist = ['banana','bareland', 'carrot','corn', 'dragonfruit','garlic','guava','peanut','pineapple',
        'pumpkin', 'rice','soybean','sugarcane','tomato']

nametoval = {}
valtoname = {}

for i in range(len(typelist)):
    nametoval[typelist[i]] =i
    valtoname[i] = typelist[i]

#filenames = []
#labels = []
file_train = []
file_test = []
label_train = []
label_test = []

for Type in typelist:
    print("reading the folder about "+ Type)
    tmp = [ './'+Type+'/'+name for name in os.listdir('./'+Type) ]
    cut = int(len(tmp)*8/10)
    if TRAIN_RATIO ==1.0:
        file_train += tmp[:cut]
        file_test += tmp[cut:]
    else:
        front = int(cut*TRAIN_RATIO)
        back = int((len(tmp)-cut)*TRAIN_RATIO)
        file_train += tmp[:front]
        file_test += tmp[-1*back:]
    label_train += [nametoval[Type]]*front
    label_test += [nametoval[Type]]*back

#import random
#c = list(zip(filenames, labels))
#random.shuffle(c)
#filenames, labels = zip(*c)


#cutline = int(len(filenames)*8/10)
print("the number of all images: ",len(file_train)+len(file_test))
print("the number of training images", len(file_train))
print("the number of training images", len(file_test))

train_dataset_size = len(file_train)
test_dataset_size = len(file_test)
#label_train = labels[:cutline]
#label_test = labels[cutline:]
#file_train = filenames[:cutline]
#file_test = filenames[cutline:]


transform =   transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = transform(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #define image 的路径
        self.images = file_train        
        #define train label
        self.target = label_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class validset(Dataset):
    def __init__(self, loader=default_loader):
        #define image 的路径
        self.images = file_test        
        #define train label
        self.target = label_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)

val_data = validset()
validloader = DataLoader(val_data, batch_size=BATCH_SIZE,shuffle =True)
#building the model

import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(typelist))
"""
model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(typelist) )

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer = SAMSGD(model.parameters(), lr=1e-1, rho=0.05)optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = SAMSGD(model_ft.parameters(), lr=1e-1, rho=0.05)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

Record = {}
Record['train_loss']= []
Record['valid_loss'] = []
Record['train_acc'] = []
Record['valid_acc'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            if phase == 'train':
                """
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                # forward
                # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                """
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    def closure():
                        optimizer.zero_grad()
                        output = model(inputs)
                        loss = criterion(output, labels)
                        loss.backward()
                        #print(loss)
                        return loss
                    loss = optimizer.step(closure)
                    #print(loss)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if phase =='val':
                for inputs, labels in validloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                

            if phase == 'train':
                scheduler.step()
            if phase == 'train':
                epoch_loss = running_loss/train_dataset_size
                epoch_acc = running_corrects.double()/train_dataset_size
            else:
                epoch_loss = running_loss/test_dataset_size
                epoch_acc = running_corrects.double()/test_dataset_size
            #紀錄
            if phase =='train':
                Record['train_loss'].append(epoch_loss)
                Record['train_acc'].append(epoch_acc.item())
            elif phase =='val':
                Record['valid_loss'].append(epoch_loss)
                Record['valid_acc'].append(epoch_acc.item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
    #print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
import time
localtime = time.localtime()
nowtime = time.strftime("%m%d-%I-%M-%p", localtime)
torch.save( model_ft.state_dict ,"./weight/"+nowtime+".pkl" )
print("model saved!")

import json
print(Record)
with open('./record/'+str(nowtime)+'.json', 'w') as f:
    json.dump(Record,f)

print("Record saved in directory: ./record")
    
