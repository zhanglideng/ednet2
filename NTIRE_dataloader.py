from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import random


class EdDataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        # self.haze_path, self.gt_path, self.depth_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        if self.gt_path:
            self.haze_data_list.sort(key=lambda x: int(x[:-4]))
            self.gt_data_list = os.listdir(self.gt_path)
            self.gt_data_list.sort(key=lambda x: int(x[:-4]))
            self.is_Gth = True
        else:
            self.haze_data_list.sort(key=lambda x: int(x[:-4]))
            self.is_Gth = False

    @staticmethod
    def random_flip(image, gt):
        """
        new_im = transforms.RandomHorizontalFlip(p=0.5)(im)  # p表示概率 水平翻转

        # 90度，180度，270度旋转

        transforms.RandomApply(transforms, p=0.5)
        功能：给一个transform加上概率，以一定的概率执行该操作

        8.随机旋转：transforms.RandomRotation
        class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
        功能：依degrees随机旋转一定角度
        参数：
        degress- (sequence or float or int) ，若为单个数，如 30，则表示在（-30，+30）之间随机旋转
        若为sequence，如(30，60)，则表示在30-60度之间随机旋转
        """
        flip = random.randint(0, 1)
        rotate = random.randint(0, 3)
        if flip == 0:
            image = transforms.RandomHorizontalFlip(p=1)(image)
            gt = transforms.RandomHorizontalFlip(p=1)(gt)
        image = transforms.RandomRotation(rotate * 90)(image)
        gt = transforms.RandomRotation(rotate * 90)(gt)
        return image, gt

    def __len__(self):
        return len(os.listdir(self.haze_path))

    def __getitem__(self, idx):

        haze_image_name = self.haze_data_list[idx]
        haze_image = cv2.imread(self.haze_path + '/' + haze_image_name)
        if self.transform1:
            haze_image = self.transform1(haze_image)
        if self.is_Gth:
            gt_image = cv2.imread(self.gt_path + '/' + haze_image_name[:-4] + '.jpg')
            if self.transform1:
                gt_image = self.transform1(gt_image)
        else:
            gt_image = False
        return self.random_flip(haze_image, gt_image)


if __name__ == '__main__':
    '''
    train_haze_path = './data/train/'
    validation_haze_path = './data/validation/'
    test_haze_path = './data/test/'
    gt_path = './data/GT'
    depth_path = './data/depth'
    path_list = [test_haze_path, gt_path, depth_path]
    print(path_list)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = Haze_Dataset(transform, path_list)
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    count = 0
    for i in dataloader:
        haze_image, gt_image, gt_depth, fog = i
        print('haze_image.shape:' + str(haze_image.shape))
        print('gt_image.shape:' + str(gt_image.shape))
        print('gt_depth.shape:' + str(gt_depth.shape))
        print('fog:' + str(fog))
        # print(hazy.shape)
        # print(gt.shape)
        count += 1
    print('count:' + str(count))
    '''

