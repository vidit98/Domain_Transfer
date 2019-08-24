from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random


class MNIST(Dataset):
    def __init__(self, file1, file2 ,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = pd.read_csv(file1, sep=" ", header=None)
        self.img.columns = ["img", "label"]
        self.img = self.img.sort_values("label")
        arr = np.array(self.img)
        self.idx = []
        for i in range(10):
        	self.idx.append(np.where(arr[:,1] == i)[0][0])


        self.root_dir = root_dir
        self.transform = transform
        self.mnist = pd.read_csv(file2, header=None)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = []
        imgp_name = []
        imgt_name = []
        batch_image = torch.zeros(5, 3, 32, 32)
        batch_imagep = torch.zeros(5, 3, 32, 32)
        batch_imaget = torch.zeros(5, 3, 32, 32)
        batch_target = torch.zeros(5)
        rows = int(self.mnist.count())
        for i in range(5):
            img_name.append(os.path.join(self.root_dir,
                                self.img.iloc[idx%(self.idx[i+1] - self.idx[i]) + self.idx[i], 0]))
            imgp_name.append(os.path.join(self.root_dir,
                                self.img.iloc[random.randint(1,self.idx[i+1] - self.idx[i]) + self.idx[i], 0]))
            k = random.randint(0,rows-1)
            imgt_name.append(self.mnist.iloc[k, 0])

        for i in range(5):
        	img = torch.from_numpy(np.array(cv2.resize(cv2.imread(img_name[i]) , (32,32)))).float()
        	imgp = torch.from_numpy(np.array(cv2.resize(cv2.imread(imgp_name[i]), (32,32)))).float()
        	
        	imgt = torch.from_numpy(np.array(cv2.resize(cv2.imread(imgt_name[i]) ,(32,32)))).float()

        	mean = torch.mean(img, (0,1))
        	meanp = torch.mean(imgp, (0,1))
        	meant = torch.mean(imgt, (0,1))

        	var = torch.var(img, (0,1))
        	varp = torch.var(imgp, (0,1))
        	vart = torch.var(imgt, (0,1))

        	transform = transforms.Compose([transforms.Normalize(mean, var)])
        	transformp = transforms.Compose([transforms.Normalize(meanp, varp)])
        	transformt = transforms.Compose([transforms.Normalize(meant, vart)])

        	img = torch.reshape(img, (3,32,32))
        	imgp =  torch.reshape(img, (3,32,32))
        	imgt =  torch.reshape(img, (3,32,32))
        	
        	img = transform(img)
        	imgp = transform(imgp)
        	imgt = transform(imgt)

        	batch_image[i] = img
        	batch_imagep[i] = imgp
        	batch_imaget[i] = imgt
        	batch_target[i] = i

        return batch_image, batch_imagep, batch_imaget, batch_target

class MNIST_Test(Dataset):
    def __init__(self, file1, file2 ,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = pd.read_csv(file1, sep=" ", header=None)
        self.img.columns = ["img", "label"]

        self.root_dir = root_dir
        self.transform = transform
        self.mnist = pd.read_csv(file2, header=None)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        imgt = 0
        img = 0
        if idx < int(self.mnist.count()):
            imgt_name = self.mnist.iloc[idx, 0]
            l = imgt_name.split("/")
            imgt = torch.from_numpy(np.array(cv2.resize(cv2.imread(imgt_name) ,(32,32)))).float()
            meant = torch.mean(imgt, (0,1))
            vart = torch.var(imgt, (0,1))
            transform = transforms.Compose([transforms.Normalize(meant, vart)])

            imgt =  torch.reshape(imgt, (3,32,32))
            imgt = transform(imgt)
            imgt = torch.reshape(imgt,(1,3,32,32) )

      
        if idx < int(self.img.count()['img']):
            img_name = self.img.iloc[idx, 0]
            img_name = os.path.join(self.root_dir,img_name)
            t = self.img.iloc[idx, 1]
            img = torch.from_numpy(np.array(cv2.resize(cv2.imread(img_name) ,(32,32)))).float()
            mean = torch.mean(img, (0,1))
            var = torch.var(img, (0,1))
            transform = transforms.Compose([transforms.Normalize(mean, var)])

            img =  torch.reshape(img, (3,32,32))
            img = transform(img)
            img = torch.reshape(img,(1,3,32,32) )

        return img, imgt, t, l[2]



