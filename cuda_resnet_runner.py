import os
import copy
import math
import sys
import copy
import json
import time
import cv2
import random
import argparse

from typing import NamedTuple

from PIL import Image
import numpy as np
import pandas as pd

from tqdm import tqdm

from collections import defaultdict

from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torch.optim.lr_scheduler
from torchvision import models

from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed


def getUTKdata(folder):
    maps={0:'caucasian',1:'afroamerican',2:'asian',3:'indian',4:'others'}
    map_gender={0:'male',1:'female'}
    images=os.listdir(folder+'UTKFace')
    allinformation=[]
    for image in images:
        if not image.startswith('.') and image != 'AIBias.ipynb':
            information=image.split('_')
            if len(information[2])!=1:
                continue
            try:
                allinformation.append(
                    {
                    'image_path':folder+'UTKFace/'+image,
                    'age':int(information[0]),# first term is age
                    'gender':map_gender[int(information[1])],#second term is gender
                    'race':maps[int(information[2])]#, #third term is race
                    })
            except Exception as e:
                print(folder+'UTKFace/'+image)
                continue
            #tx:the information we get from parsing the entry name are: age, gender and race in a human readable form [{image_path: ./././ age: 59 gender: female race: indian}..]
    return allinformation


class Img_Dataset(Dataset):
    '''
     imgs: numpy array, 0-255
     img_size: width*height
    '''
    def __init__(self, imgs ,img_size, labels = None, transform = None):

        self.transform = transform
        self.labels = labels
        self.total_num = imgs.shape[0]
        def load_image(idx):
            cur_img_arr = imgs[idx]
            img = Image.fromarray(np.uint8(cur_img_arr))
            return img.resize(img_size)
        self.load_image = load_image

    def __getitem__(self,index):
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            label = 0
        else:
            label = self.labels[index]
        return img, label  # 0 is the class

    def __len__(self):
        return self.total_num


class Img_Dataset_Iter(Dataset):
    '''
     imgs: numpy array, 0-255
     img_size: width*height
    '''
    def __init__(self, img_paths ,img_size, labels = None, transform = None):

        self.transform = transform
        self.labels = labels
        self.total_num = len(img_paths)
        def load_image(idx):
            cur_img_arr = img_paths[idx]
            img = Image.open(cur_img_arr)
            if img.mode=='L':
              img=img.convert("RGB")
            newimg=img.resize(img_size)
            img.load()
            return newimg
        self.load_image = load_image

    def __getitem__(self,index):
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            label = 0
        else:
            label = int(self.labels[index])
        return img, label  # 0 is the class

    def __len__(self):
        return self.total_num


def get_min_max_sample(num_sample):
    max_sample = np.inf
    min_sample = 0
    # the key below is race, find the a common region of age that all races are populated
    for key in num_sample:
        max_sample = min(max_sample,np.quantile(list(num_sample[key].values()),0.8))
        min_sample = max(min_sample,np.quantile(list(num_sample[key].values()),0.2))
    return min_sample, max_sample


def update(select_size, threshold, num,ds_num):
    threshold -= num
    ds_num -= 1
    if ds_num!=0:
        select_size = math.ceil(threshold*1.0/ds_num)
    else:
        select_size = threshold
    return select_size, threshold, ds_num


def make_dataloader_iter(samples,labels,img_size=(32,32), batch_size=256, transform_test=None, shuffle=False, num_workers=0):
    dataset = Img_Dataset_Iter(samples,labels = labels,img_size = img_size, transform=transform_test)
    print("dataset:", dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader


def process_data(path):
    X,y=[],[]
    f=open(path,'r')
    lines=f.readlines()
    classes=set()
    for line in lines:
        temp=line.strip()
        if temp is not None:
            X.append(temp.split('\t')[0])
            y.append(int(temp.split('\t')[2]))
            classes.add(temp.split('\t')[2])
    # this should return num_classes but it is wrong
    return X,y,101


def process_data_ood(path,gender=False):
    X,y=[],[]
    f=open(path,'r')
    lines=f.readlines()
    classes=set()
    genderMap={
        'male':0,
        'female':1
    }
    race=[]
    for line in lines:
        temp=line.strip()
        if temp is not None:
            if len(temp.split('\t'))<4:continue
            if not os.path.exists(temp.split('\t')[0]):continue
            X.append(temp.split('\t')[0])

            y.append(int(temp.split('\t')[2]))
            classes.add(temp.split('\t')[2])
            race.append(temp.split('\t')[1])
    return X,y,len(classes),race


def flip_image(image_path):
    im = Image.open(image_path)
    im_flipped = im.transpose(method=Image.FLIP_LEFT_RIGHT)
    save_path = image_path + '_flip.jpg'
    im_flipped.save(save_path, 'JPEG')
    return save_path


def get_balanced_data(data_folder, train_save_path='./data/train_new.tsv', test_save_path='./data/test_new.tsv'):

    dataset_names = ['UTKdata']
    races = ['caucasian', 'afroamerican', 'asian']
    all_datasets = {
        'UTKdata': getUTKdata(data_folder),
    }
    num_samples_tmp = {
        race: {i: 0 for i in dataset_names} for race in races
    }
    dataset_samples = {
        i: copy.deepcopy(num_samples_tmp) for i in range(0, 101)
    }
    all_samples_tmp = {
        race: defaultdict(list) for race in races
    }
    all_samples = {
        dataset: copy.deepcopy(all_samples_tmp) for dataset in dataset_names
    }
    num_sample = {
        race: {i: 0 for i in range(0, 101)} for race in races
    }
    for dataset in all_datasets:
        for samples in tqdm(all_datasets[dataset]):
            if 0 <= samples['age'] <= 100 and samples['race'] in ['caucasian', 'afroamerican', 'asian']:
                file_path = samples['image_path'].replace('OriDatasets', 'AliDatasets_new')
                if not os.path.exists(file_path):
                    continue
                all_samples[dataset][samples['race']][samples['age']].append(
                    [file_path, samples['race'], samples['age']])
                dataset_samples[samples['age']][samples['race']][dataset] += 1
                num_sample[samples['race']][samples['age']] += 1
                try:
                    save_path = flip_image(file_path)
                except Exception as e:
                    print(file_path, e)
                    continue
                all_samples[dataset][samples['race']][samples['age']].append(
                    [save_path, samples['race'], samples['age']])
                dataset_samples[samples['age']][samples['race']][dataset] += 1
                num_sample[samples['race']][samples['age']] += 1
    for key in num_sample:
        samples = copy.deepcopy(num_sample[key])
        num_sample[key] = dict(sorted(samples.items(), key=lambda samples: samples[1]))
    min_sample, max_sample = get_min_max_sample(num_sample)

    balanced_train_data = []
    balanced_test_data = []
    train_data_num = {
        race: {i: 0 for i in range(0, 101)} for race in races
    }

    for age in range(1, 101):

        threshold = np.inf
        for race in num_sample:
            threshold = min(threshold, num_sample[race][age])
        threshold = int(min(max_sample, max(min_sample, threshold)))

        ds_num = len(all_datasets)
        select_size = math.ceil(threshold * 1.0 / ds_num)

        for race in num_sample:
            race_threshold = threshold
            race_select_size = select_size
            ds_num = len(all_datasets)

            race_num_sample = copy.deepcopy(dataset_samples[age][race])
            dataset_samples[age][race] = dict(
                sorted(race_num_sample.items(), key=lambda race_num_sample: race_num_sample[1]))

            for dataset in dataset_samples[age][race]:
                num = dataset_samples[age][race][dataset]
                if num > race_select_size:
                    train_size = math.ceil(num * 0.8)
                    for index in range(train_size):
                        balanced_train_data.append(all_samples[dataset][race][age][index])
                        train_data_num[race][age] += 1
                    for index in range(train_size, num):
                        balanced_test_data.append(all_samples[dataset][race][age][index])
                    race_select_size, race_threshold, ds_num = update(race_select_size, race_threshold, num, ds_num)
                else:

                    race_select_size = len(all_samples[dataset][race][age])
                    indices = np.random.choice(len(all_samples[dataset][race][age]), race_select_size, replace=False)
                    train_size = math.floor(race_select_size * 0.8)
                    ds_num -= 1
                    for index in range(train_size):
                        balanced_train_data.append(all_samples[dataset][race][age][indices[index]])
                        train_data_num[race][age] += 1
                    for index in range(train_size, len(indices)):
                        balanced_test_data.append(all_samples[dataset][race][age][indices[index]])

    balanced_test_data = pd.DataFrame(balanced_test_data)
    balanced_train_data = pd.DataFrame(balanced_train_data)
    balanced_test_data.to_csv(test_save_path, header=None, index=None, sep='\t')
    balanced_train_data.to_csv(train_save_path, header=None, index=None, sep='\t')
    return


def get_separate_data(file_path):
    print(file_path)
    f = open(file_path, 'r')
    test_train = file_path.split('/')[-1].split('_')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    lines = f.readlines()
    goals = defaultdict(list)
    for line in lines:
        goal = line.strip().split('\t')[0].split('/')[3]
        goals[goal].append(line)

    for keys in goals:
        f = open('{}/{}_{}.tsv'.format(folder, keys, test_train), 'w')
        for i in goals[keys]:
            f.write(i)


def adjust_opt(optAlg, optimizer, epoch,lr):
    if optAlg == 'sgd':
        if epoch % 10==0: lr = lr*(0.1**(epoch//10))
        else: return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if optAlg == 'adam':
        if epoch % 40==0: lr = lr*(0.2**(epoch//40))
        else: return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_baseline(net,train_loader,optimizer,state, device):
    net.train()

    loss_avg = 0.0
    correct  = 0

    total = len(train_loader)
    widgets = ['Training: ',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
    progress = ProgressBar(widgets=widgets, maxval=total)
    mae=0.0
    # Begin training
    for data, target in progress(train_loader):
        data, target = data.to(device), target.to(device)


        # forward
        output = net(data)
        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss.data) * 0.2

        # accuracy
        pred = output.data.max(1)[1]
#         print("OUTPUT --------------")
#         print(output.data.max(1)[1])
#         print("TARGET --------------")
#         print(target.data)

        correct += pred.eq(target.data).sum().item()
        outputs=F.softmax(output,dim=1)

        for i in range(len(output)):
            age=float(expected_age(outputs[i].data.to(device)))
            mae+=abs(age-float(target[i].data.to(device)))

    progress.finish()
    state['train_loss'] = loss_avg
    state['train_accuracy'] = correct / len(train_loader.dataset)
    state['train_mae']=mae/len(train_loader.dataset)
    print("train_loss:{},train_accuracy;{};train_mae{}".format(state['train_loss'], state['train_accuracy'], state['train_mae']))


def expected_age(vector):
    # Get expected age accoring to probabilities and ages
    # Used for DEX

    res = [(i)*v for i, v in enumerate(vector)]
    # print(vector,sum(res))
    return sum(res)


# test function
def test(net,test_loader,state,device):
    # Enter evaluzaion mode
    net.eval()

    # Recording test results
    loss_avg = 0.0
    correct = 0

    # Visualizing test procedure
    total = len(test_loader)
    widgets = ['Testing: ',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
    progress = ProgressBar(widgets=widgets, maxval=total)

    mae=0.0

    with torch.no_grad():
        for data, target in progress(test_loader):
            #data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            outputs=F.softmax(output,dim=1)

            # mae
            for i in range(len(pred)):
                age=float(expected_age(outputs[i].data.to(device)))
                # break
                mae+=abs(age-float(target[i].data.to(device)))
            # break
            # test loss average
            loss_avg += float(loss.data)

    progress.finish()
    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)
    state['test_mae']=mae/len(test_loader.dataset)
    print("\ntest_loss:{},test_accuracy;{},test_mae:{}".format(state['test_loss'], state['test_accuracy'],state['test_mae']))
#Generalmodels.py

def alexnet(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model=models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    else:
        model=models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    return model

def VGG16(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model=models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    else:
        print('Using both')
        model=models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 101)
        #tx: file is not pickled. so no need to unpickle it
#        model.load_state_dict(torch.load(trained_model))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        print(model.classifier[6])
    return model

def densenet121(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
    else:
        model=models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        print('Using trained model from {}'.format(trained_model))
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
    print(model.classifier)
    return model

#tx: resnet50 is a 50-layer deep CNN
def resnet50(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    else:
        model=models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        print('Using trained model from {}'.format(trained_model))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    return model

def create_Resnet(resnet_number=50, num_classes=100,
                  gaussian_layer=True, weight_path=None,
                  device=None):
    model = getattr(models, 'resnet'+str(resnet_number))()
    if weight_path is not None:
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        state = torch.load(weight_path)
        model.load_state_dict(state)
    if gaussian_layer:
        model.fc = GaussianLayer(2048, n_classes=num_classes)
    if device is not None:
        model.to(device)
    return model
#train.py

def train_model(train_path, test_path, batch_size=32, model_name = "resnet50", opt="sgd",dataset="UTKFace",num_epochs=100,lr=0.01,
                pretrain=False,trained_model=None, device_type="cpu"):
    # Configuration
    state = defaultdict()
    opt = opt
    img_pixels=(200,200)

    # loading data
    test_set_X,test_set_y,_= process_data(test_path)
    train_set_X,train_set_y,num_classes=process_data(train_path)
    train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    train_loader = make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels, batch_size=batch_size,
                                   transform_test=train_transform, shuffle=True)
    #train_loader
    test_loader = make_dataloader_iter(test_set_X, test_set_y, img_size=img_pixels,
                                batch_size=batch_size, transform_test=test_transform)

    # For pre-train and fine-tune
    if pretrain:
        train_set_X,train_set_y,num_classes= process_data(train_path)
        save_path = "./model_weights/pretrained/{}_{}_{}".format(model_name,opt,str(lr))
    else:
        num_classes=101
        save_path = "./model_weights/{}_{}_{}_{}".format(model_name,dataset,opt,str(lr))
    model_type = "{}_{}".format(model_name, dataset)
    print('Using Data: ',train_path)

    # Initializating model
    print("num_classes: ",num_classes)
    if model_name=='VGG':
        net=VGG16(num_classes,pretrain, trained_model,if_test=False)
        print("get VGG")
    elif model_name=='resnet50':
        net = resnet50(num_classes,pretrain, trained_model,if_test=False)
    elif model_name=='densenet121':
        net = densenet121(num_classes,pretrain, trained_model,if_test=False)
    elif model_name=='alexnet':
        net = alexnet(num_classes,pretrain, trained_model,if_test=False)

    # Set device
    device=torch.device(device_type)
    net=net.to(device)

    # Defining opt method
    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Create results saving path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)
    with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'w') as f:
        f.write('epoch,time(s),train_loss,train_acc,train_mae,test_loss,test_acc(%),test_mae\n')
    print('Beginning Training for {} on {}\n'.format(model_name,dataset))
    # Main loop
    best_epoch = 100
    best_acc = 0.0
    best_mae = 100.0
    prev_path=' '
    for epoch in range(num_epochs):
        adjust_opt(opt, optimizer, epoch,lr)
        state['epoch'] = epoch
        begin_epoch = time.time()

        # Train and Test
        train_baseline(net,train_loader,optimizer,state,device)
        test(net,test_loader,state,device)
        scheduler.step()
         # Save model
        cur_mae = state['test_mae']
        if cur_mae < best_mae:
            cur_save_path = os.path.join(save_path, '{}_epoch_{}_{}.pt'.format(model_type,epoch,cur_mae))
            torch.save(net.state_dict(),cur_save_path)
            if os.path.exists(prev_path):
                os.remove(prev_path)
            prev_path = cur_save_path
            best_epoch = epoch
            best_mae = cur_mae

        # Save results
        with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.4f,%0.4f,%0.6f,%0.4f,%0.4f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['train_accuracy'],
                state['train_mae'],
                state['test_loss'],
                state['test_accuracy'],
                state['test_mae']
            ))

        # Print results
        print('|Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f}|  Train Acc {3:.4f} | Test Loss {4:.4f} | Test Acc {5:.4f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['train_accuracy'],
        state['test_loss'],
        state['test_accuracy'])
        )
# Removes flipped images generated from data pre-process
def clean():
    UTKFace_path = './data/UTKFace'
    original_path = './data/original'
    data_path = './data'
    for filename in os.listdir(UTKFace_path):
        if filename.endswith(".jpg_flip.jpg"):
            file_path = os.path.join(UTKFace_path, filename)
            os.remove(file_path)
    for filename in os.listdir(original_path):
        if filename.endswith("jpg_train.tsv") or filename.endswith("jpg_test.tsv"):
            file_path = os.path.join(original_path, filename)
            os.remove(file_path)
    file1 = open("./data/test_new.tsv", "r+")
    file1.truncate(0)
    file1.close()
    file2 = open("./data/train_new.tsv", "r+")
    file2.truncate(0)
    file2.close()

# Assumes you are in the /AIBias/ directory
data_folder = './data/'
train_save_path = './data/train_new.tsv'
test_save_path = './data/test_new.tsv'
clean()

get_balanced_data(data_folder, train_save_path, test_save_path)
get_separate_data(train_save_path)
get_separate_data(test_save_path)

datafolder = './data'
train_path = 'train_new.tsv'
test_path = 'test_new.tsv'
train_path = os.path.join(datafolder, train_path)
test_path = os.path.join(datafolder, test_path)

model_name = 'resnet50'
dataset = 'UTKFace'
opt = "adam"
lr=0.0001
num_epoches=10
pretrain="True"
trained_model="./data/train_resnet50"
device_type="cuda"

# Checking the existance of pre-trained model
if not pretrain:
    if not os.path.exists(trained_model):
        raise FileExistsError("Pretrained Model does not exist!")
    if trained_model.split('/')[-1].split('_')[1]!=model_name:
        raise ValueError("Model name does not match the pre-trained model!")
lists=os.listdir(trained_model)

# Getting the path of the pre-trained model
for i in lists:
    if i.split('.')[-1]=='pt':
        trained_model=os.path.join(trained_model,i)
print("trained_model:", trained_model)
print("opt:", opt)

# Train
train_model(train_path, test_path, model_name=model_name,
opt=opt,dataset=dataset,num_epochs=num_epoches,lr=lr,pretrain=pretrain,trained_model=trained_model, device_type=device_type)
