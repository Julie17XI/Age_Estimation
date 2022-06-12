import copy
import math
import os
import time
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as trn
import torch.optim.lr_scheduler
from PIL import Image
from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed
from torchvision import models
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

#
#
# Data Preprocessing
#
#

#
# Get image information from datasets
#

# UTKFace image has a title like this:
# [age]_[gender]_[race]_[date&time].jpg
# example: 1_0_0_20161219140623097.jpg
# age is an integer from 0 - 116
# gender is either 0 (male) or 1 (female)
# race is an integer from 0 - 4, denoting caucasian, afroamerican, asian, indian and others
def get_UTKdata(dataset_folder):
    '''Collect directory, race, gender, age information of each image in UTKFace'''
    map_race={0:'caucasian',1:'afroamerican',2:'asian',3:'indian',4:'others'}
    map_gender={0:'male',1:'female'}
    images=os.listdir(dataset_folder+'UTKFace')
    all_information=[]
    for image in images:
        if not image.startswith('.'):
            filename_info=image.split('_')
            try:
                all_information.append(
                    {
                    'image_path':dataset_folder+'UTKFace/'+image,
                    'age':int(filename_info[0]),
                    'gender':map_gender[int(filename_info[1])],
                    'race':map_race[int(filename_info[2])]
                    })
            except Exception as e:
                print(dataset_folder+'UTKFace/'+image, e)
                continue
    return all_information


def flip_image(image_path):
    '''Flip the image from benchmark datasets'''
    im = Image.open(image_path)
    im_flipped = im.transpose(method=Image.FLIP_LEFT_RIGHT)
    save_path = image_path + '_flip.jpg'
    im_flipped.save(save_path, 'JPEG')
    return save_path

#
# Make the percentage of each race for each age to be equivalent
#

def get_min_max_sample(race_age_counter):
    '''Decide the common range of data size for all ages and for all considered races'''
    # If the 20 percentile of caucasian among all ages is 800, and the 80 percentile is 100
    # and the 20 percentile of afroamerican among all ages is 700, and the 80 percentile is 90
    # and the 20 percentile of asian among all ages is 1000, and the 80 percentile is 200
    # the max_sample will be 700
    # the min_sample will be 200
    max_sample = np.inf
    min_sample = 0
    for race in race_age_counter:
        max_sample = min(max_sample,np.quantile(list(race_age_counter[race].values()),0.8))
        min_sample = max(min_sample,np.quantile(list(race_age_counter[race].values()),0.2))
    return min_sample, max_sample

def update(race_select_size, race_threshold, dataset_sample_size, benchmark_datasets_num): # not using it
    race_threshold -= dataset_sample_size
    benchmark_datasets_num -= 1
    if benchmark_datasets_num!=0:
        race_select_size = math.ceil(race_threshold*1.0/benchmark_datasets_num)
    else:
        race_select_size = race_threshold
    return race_select_size, race_threshold, benchmark_datasets_num

def get_balanced_data(data_folder, train_save_path='./data/train_new.tsv', test_save_path='./data/test_new.tsv'):
    '''Make the percentage of each race for each age to be equivalent'''
    # Read from benchmark datasets
    benchmark_datasets = ['UTKdata']
    races = ['caucasian', 'afroamerican', 'asian']
    benchmark_datasets_data = {
        'UTKdata': get_UTKdata(data_folder),
    }

    # Store the data from every benchmark datasets to dataset_race_age_information dictionary
    race_age_information = {
        race: defaultdict(list) for race in races
    }
    dataset_race_age_information = {
        dataset: copy.deepcopy(race_age_information) for dataset in benchmark_datasets
    }


    race_dataset_counter = {
        race: {i: 0 for i in benchmark_datasets} for race in races
    }
    age_race_dataset_counter = {
        i: copy.deepcopy(race_dataset_counter) for i in range(0, 101)
    }

    race_age_counter = {
        race: {i: 0 for i in range(0, 101)} for race in races
    }

    for dataset in benchmark_datasets_data:
        for data in tqdm(benchmark_datasets_data[dataset]):
            if 0 <= data['age'] <= 100 and data['race'] in ['caucasian', 'afroamerican', 'asian']:
                file_path = data['image_path']
                if not os.path.exists(file_path):
                    continue
                dataset_race_age_information[dataset][data['race']][data['age']].append(
                    [file_path, data['race'], data['age']])
                age_race_dataset_counter[data['age']][data['race']][dataset] += 1
                race_age_counter[data['race']][data['age']] += 1
                try:
                    save_path = flip_image(file_path)
                except Exception as e:
                    print(file_path, e)
                    continue
                dataset_race_age_information[dataset][data['race']][data['age']].append(
                    [save_path, data['race'], data['age']])
                age_race_dataset_counter[data['age']][data['race']][dataset] += 1
                race_age_counter[data['race']][data['age']] += 1

    for race in race_age_counter:
        age_counter = copy.deepcopy(race_age_counter[race])
        race_age_counter[race] = dict(sorted(age_counter.items(), key=lambda age_counter: age_counter[1])) # Order age by the numbder of data it has
    min_sample, max_sample = get_min_max_sample(race_age_counter)

    balanced_train_data = []
    balanced_test_data = []

    # Find how many samples we should select from every dataset for the given age
    for age in range(1, 101):
        smallest_sample_size = np.inf
        for race in race_age_counter:
            smallest_sample_size = min(smallest_sample_size, race_age_counter[race][age]) # threshold is the smallest sample size among all races for the given age
        threshold = int(min(max_sample, max(min_sample, smallest_sample_size)))
        # If for age = 34, the sample sizes for caucasian, afroamerican, asian are 1200, 800, 900
        # and the max_sample is 700, the min_sample is 200
        # the smallest sample size should be 800
        # the threshold should be 700

        # If for age = 99, the sample sizes for caucasian, afroamerican, asian are 2, 1, 2
        # and the max_sample is 700, the min_sample is 200
        # the smallest sample size should be 1
        # the threshold should be 200

        # If for age = 80, the sample sizes for caucasian, afroamerican, asian are 600, 600, 650
        # and the max_sample is 700, the min_sample is 200
        # the smallest sample size should be 600
        # the threshold should be 600

        benchmark_datasets_num = len(benchmark_datasets_data)
        select_size = math.ceil(threshold * 1.0 / benchmark_datasets_num)

        for race in race_age_counter:
            race_threshold = threshold
            race_select_size = select_size

            race_race_age_counter = copy.deepcopy(age_race_dataset_counter[age][race])
            age_race_dataset_counter[age][race] = dict(
                sorted(race_race_age_counter.items(), key=lambda race_race_age_counter: race_race_age_counter[1])) # Sort age_race_dataset_counter by the number of people, not useful when we only have one benchmard dataset

            for dataset in age_race_dataset_counter[age][race]:
                dataset_sample_size = age_race_dataset_counter[age][race][dataset]
                if dataset_sample_size > race_select_size:
                    train_size = math.ceil(dataset_sample_size * 0.8)
                    for index in range(train_size):
                        balanced_train_data.append(dataset_race_age_information[dataset][race][age][index])
                    for index in range(train_size, dataset_sample_size):
                        balanced_test_data.append(dataset_race_age_information[dataset][race][age][index])
                    #race_select_size, race_threshold, benchmark_datasets_num = update(race_select_size, race_threshold, dataset_sample_size, benchmark_datasets_num)
                else:
                    race_select_size = dataset_sample_size
                    #indices = np.random.choice(len(dataset_race_age_information[dataset][race][age]), race_select_size, replace=False)
                    train_size = math.floor(dataset_sample_size * 0.8)
                    #benchmark_datasets_num -= 1
                    for index in range(train_size):
                        balanced_train_data.append(dataset_race_age_information[dataset][race][age][index])
                    for index in range(train_size, dataset_sample_size):
                        balanced_test_data.append(dataset_race_age_information[dataset][race][age][index])
    return balanced_test_data, balanced_train_data

#
# Do all the processing on custom dataset
#
def resize_custom_image(image_path):
    pass

def flip_custom_image(image_path):
    '''Flip an image and save to same directory'''
    im = Image.open(image_path)
    im_flipped = im.transpose(method=Image.FLIP_LEFT_RIGHT)
    save_path = image_path + '_flip.png'
    im_flipped.save(save_path, 'PNG')
    return save_path

def argument_custom_dataset(dataset_dir):
    '''Flip all images'''
    for file_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, file_name)
        try:
            flip_custom_image(image_path)
        except Exception as e:
            print(image_path, e)
            continue

def get_custom_dataset_age_dictionary(dataset_dir):
    '''Use the dictionary to save info of images with the same age'''
    # Read the title of each image of the custom dataset
    # Save the image information to a dictionary
    # The key value pair in the dictionary looks like this:
    # age:[image1_info, image2_info]
    # The value of each key value pair looks like this:
    # ['directory of the facial image', 'race', 'age]
    # The dictionary looks like this:
    # dict = {0: ['./data/OneIndividual/0_512_StevenOHara.png', 'caucasian', 0]}
    age_dict = {}
    for file_name in os.listdir(dataset_dir):
        age = file_name.split('_')[0]
        image_path = os.path.join(dataset_dir, file_name)
        if age in age_dict:
            age_dict[age].append([image_path, 'caucasian', age])
        else:
            age_dict[age] = [[image_path, 'caucasian', age]]
    return age_dict


def get_supplementary_data(age_dict, train_save_path='./data/train_new.tsv', test_save_path='./data/test_new.tsv'):
    '''Randomly save 80% images to training set, and 20% to test set'''
    # Read custom dataset age dictionary
    # For each age, randomly choose 80% of samples to save in balanced_train_data
    # For each age, save the rest of the sample to balanced_test_data
    for age in age_dict:
        num_images = len(age_dict[age])
        items = age_dict[age]
        train_size = math.floor(num_images * 0.8)
        test_size = num_images - train_size
        i = train_size
        while i > 0:
            index = random.randint(0, i)
            item = items.pop(index)
            balanced_train_data.append(item)
            i -= 1
        print(f"rest = {age}")
        for item in items:
            balanced_test_data.append(item)
    return


def load_custom_data(dataset_dir):
    '''Argument custom dataset and load data to training set and test set'''
    argument_custom_dataset(dataset_dir)
    age_dict = get_custom_dataset_age_dictionary(dataset_dir)
    get_supplementary_data(age_dict)


def convert_data_to_tabular(balanced_train_data, balanced_test_data, train_save_path='./data/train_new.tsv', test_save_path='./data/test_new.tsv'):
    balanced_test_data = pd.DataFrame(balanced_test_data)
    balanced_train_data = pd.DataFrame(balanced_train_data)
    balanced_test_data.to_csv(test_save_path, header=None, index=None, sep='\t')
    balanced_train_data.to_csv(train_save_path, header=None, index=None, sep='\t')
    return

#
#
# Training and Testing
#
#

#
# General models
#

def alexnet(classes, pretrain, trained_model, if_test=False):
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

def VGG16(classes, pretrain, trained_model, if_test=False):
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
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        print(model.classifier[6])
    return model

def densenet121(classes, pretrain, trained_model, if_test=False):
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

def resnet50(classes, pretrain, trained_model, if_test=False):
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


def train_baseline(net,train_loader, optimizer, state, device):
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


def process_data(path):
    X,y=[],[]
    f=open(path,'r')
    lines=f.readlines()
    for line in lines:
        temp=line.strip()
        if temp is not None:
            X.append(temp.split('\t')[0])
            y.append(int(temp.split('\t')[2]))
    return X,y

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
        return img, label

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
        return img, label

    def __len__(self):
        return self.total_num


def make_dataloader_iter(samples,labels,img_size=(32,32), batch_size=256, transform_test=None, shuffle=False, num_workers=0):
    dataset = Img_Dataset_Iter(samples,labels = labels,img_size = img_size, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

def train_model(train_path, test_path, batch_size=32, model_name = "resnet50", opt="sgd",dataset="UTKFace",num_epochs=100,lr=0.01,
                pretrain=False,trained_model=None, device_type="cpu"):
    # Configuration
    state = defaultdict()
    opt = opt
    img_pixels=(200,200)
    num_classes=101
    # loading data
    test_set_X,test_set_y = process_data(test_path)
    train_set_X,train_set_y = process_data(train_path)
    train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    train_loader = make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels, batch_size=batch_size,
                                        transform_test=train_transform, shuffle=True)
    #train_loader
    test_loader = make_dataloader_iter(test_set_X, test_set_y, img_size=img_pixels, batch_size=batch_size,
                                        transform_test=test_transform)

    # For pre-train and fine-tune
    if pretrain:
        train_set_X,train_set_y= process_data(train_path)
        save_path = "./model_weights/pretrained/{}_{}_{}".format(model_name,opt,str(lr))
    else:
        save_path = "./model_weights/{}_{}_{}_{}".format(model_name,dataset,opt,str(lr))
    model_type = "{}_{}".format(model_name, dataset)
    print('Using Data: ',train_path)

    # Initializating model
    print("num_classes: ",num_classes)
    if model_name=='VGG':
        net=VGG16(num_classes, pretrain, trained_model, if_test=False)
        print("get VGG")
    elif model_name=='resnet50':
        net = resnet50(num_classes, pretrain, trained_model, if_test=False)
    elif model_name=='densenet121':
        net = densenet121(num_classes, pretrain, trained_model, if_test=False)
    elif model_name=='alexnet':
        net = alexnet(num_classes, pretrain, trained_model, if_test=False)

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
#
# Removes data generated from data pre-process
#
def clean():
    '''Remove flipped images and tsv entries'''
    UTKFace_path = './data/UTKFace'
    custom_path = './data/OneIndividual'
    original_path = './data/original'
    for filename in os.listdir(UTKFace_path):
        if filename.endswith(".jpg_flip.jpg"):
            file_path = os.path.join(UTKFace_path, filename)
            os.remove(file_path)
    for filename in os.listdir(custom_path):
        if filename.endswith(".png_flip.png"):
            file_path = os.path.join(custom_path, filename)
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

#
# Start of main
#

# Assumes you are in the project root directory
dataset_folder = './data/'
custom_dataset_path = './data/OneIndividual/'
train_save_path = './data/train_new.tsv'
test_save_path = './data/test_new.tsv'
clean()

#
balanced_test_data, balanced_train_data = get_balanced_data(dataset_folder, train_save_path, test_save_path)
# argument_custom_dataset(custom_dataset_path)
# dict = get_custom_dataset_age_dictionary(custom_dataset_path)
# print(f"age_dict: {dict}")
# load_custom_data(custom_dataset_path)
convert_data_to_tabular(balanced_train_data, balanced_test_data, )
#get_balanced_data(data_folder, train_save_path, test_save_path)

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
