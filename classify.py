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

print("Batch size: ", BATCH_SIZE)

typelist = ['banana','bareland', 'carrot','corn', 'dragonfruit','garlic','guava','peanut','pineapple','pumpkin', 'rice','soybean','sugarcane','tomato']

nametoval = {}
valtoname = {}
for i in range(len(typelist)):
    valtoname[i] = typelist[i]

test_folders ='test_0 test_1 test_2 test_3 test_4 test_5 test_6 test_7 test_8 test_9 test_a test_b test_c test_d test_e test_f'.split(' ')

#test_folders = ['test_0']
file_test = []

for folder in test_folders:
    print("reading the folder in "+folder)
    tmp = [ os.path.join('./','test_data' ,folder,name) for name in os.listdir('./test_data/'+folder) ]
    print('loading ',len(tmp),' images' )
    file_test +=tmp
print('totally loading ', len(file_test),' images')

transform =   transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = transform(img_pil)
    return img_tensor


class testset(Dataset):
    def __init__(self, loader=default_loader):
        #define image 的路径
        self.images = file_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        #keep this in mind
        return img, fn

    def __len__(self):
        return len(self.images)

test_data  = testset()
testloader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False)
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(typelist) )
model.load_state_dict(torch.load('./weight/0515-09-50-PM.pkl'))
model = model.to(device)

model.eval()

import csv
count = 0
with open('for_upload.csv', 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["image_filename", "label"])
    for inputs,name in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #print(name, preds.cpu().numpy())
        for a, b in zip(name, preds.cpu().numpy()):
            writer.writerow([a.split('/')[-1], valtoname[b]])
        #break
        count +=1
        print(count)
print('down')
