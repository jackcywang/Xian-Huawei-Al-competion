# coding=utf-8
import os
import os.path as osp
import random
import codecs
from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
# from RandAugment import RandAugment

class Xian_Dataset(Dataset):
    def __init__(self,args,img_paths, labels, mode=None):
        assert mode == 'train' or mode == 'val'
        self.img_paths = img_paths
        self.labels= labels
        self.mode = mode

        if self.mode == 'train':
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(args.imgsize,scale=(0.4,1.5)),
                    random_erase(),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ]
            )
        #    self.transform.transforms.insert(0,RandAugment(1,13))

        elif self.mode == 'val':
            self.transform = transforms.Compose(
                [   
                    transforms.Resize(args.imgsize),
                    transforms.CenterCrop(args.imgsize),
                    transforms.ToTensor(),  # 2 tens
                    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # 标准化               
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) 
        return img,label


class Multi_scale(Dataset):
    def __init__(self,args,img_paths, labels, epoch, mode=None):
        assert mode == 'train' or mode == 'val'
        self.img_paths = img_paths
        self.labels= labels
        self.mode = mode
        if epoch < args.multi_milestone[0]:
            args.imgsize = args.multi_size[0]
            args.batch_size = 48
        elif epoch < args.multi_milestone[1]:
            args.imgsize = 24
            args.batch_size = args.batch_size/2
        elif epoch < args.total_epoch
            args.imgsize = args.multi_size[2]
            args.batch_size = 12

        if self.mode == 'train':
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(args.imgsize,scale=(0.4,1.5)),
                    random_erase(),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ]
            )
        #    self.transform.transforms.insert(0,RandAugment(1,13))

        elif self.mode == 'val':
            self.transform = transforms.Compose(
                [   
                    transforms.Resize(args.imgsize),
                    transforms.CenterCrop(args.imgsize),
                    transforms.ToTensor(),  # 2 tens
                    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # 标准化               
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) 
        return img,label


class Pseudo_Dataset(Dataset):
    def __init__(self,args,img_paths, labels):
        self.img_paths = img_paths
        self.labels= labels
        self.transform = transforms.Compose(
            [   
                transforms.Resize(args.imgsize),
                transforms.CenterCrop(args.imgsize),
                transforms.ToTensor(),  # 2 tens
                transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # 标准化               
            ]
        )
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) 
        return img,label

 

class random_crop(object):
    def __init__(self, re_size=450, crop_size=380, p=0.5):
        '''
        以一定概率随机剪切
        :param re_size: 先调整到这个尺寸
        :param crop_size: 剪切尺寸
        :param p: 执行随机剪切的概率
        '''
        self.resize_big = transforms.Resize(re_size)
        self.resize_small = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
        ])
        self.crop = transforms.RandomCrop(crop_size)
        self.p = p

    def __call__(self, img):
        # 执行剪切
        if random.random()<=self.p:
            # print('crop')
            img = self.resize_big(img)
            return self.crop(img)
        # resize 返回
        return self.resize_small(img)


class random_erase(object):
    def __init__(self, erase_size=48, p=0.7):
        self.erase_size = erase_size
        self.p = p

    def __call__(self, img):
        if random.random()<self.p:
            # print('erase')
            w,h = img.size
            start_x = random.randint(0, w-self.erase_size-1)
            start_y = random.randint(0, h-self.erase_size-1)

            erase_np = np.random.random((self.erase_size,self.erase_size,3))
            erase_img = Image.fromarray(erase_np, mode='RGB')
            img.paste(erase_img,(start_x,start_y,start_x+self.erase_size,start_y+self.erase_size))
        return img


class Cutout(object):
    
    def __init__(self, n_holes, length,p):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
            Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """  
  
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        if random.random() < self.p:
            img = img * mask 
        else:
            img = img          

        return img

def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


def get_data(args,mode):
    if mode == 'train':
        dataset_path = args.data_train
    else:
        dataset_path = args.data_val
    images_list = []
    labels_list = []
    samples_each_class = [0]*54
    classes = os.listdir(dataset_path)
    for i in classes:
        lb = int(i)
        class_path = osp.join(dataset_path,i)
        samples_each_class[lb] = len(os.listdir(class_path))
    for root, parent, names in os.walk(dataset_path):
        for name in names:
            img_path = osp.join(root,name)
            label = img_path.split('/')[-2]
            label = int(label)
            images_list.append(img_path)
            labels_list.append(label)
    return images_list,labels_list,samples_each_class



