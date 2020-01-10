from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


class EdDataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        # self.haze_path, self.gt_path, self.depth_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        if self.gt_path:
            self.haze_data_list.sort(key=lambda x: float(x[-8:-4]))
            self.haze_data_list.sort(key=lambda x: int(x[:-30]))
            self.gt_data_list = os.listdir(self.gt_path)
            self.gt_data_list.sort(key=lambda x: int(x[:-4]))
            self.is_Gth = True
        else:
            self.haze_data_list.sort(key=lambda x: int(x[:-4]))
            self.is_Gth = False

    def __len__(self):
        return len(os.listdir(self.haze_path))

    def __getitem__(self, idx):
        """
            需要传递的信息有：
            有雾图像
            无雾图像
            (深度图)
            (雾度)
            (大气光)
        """
        haze_image_name = self.haze_data_list[idx]
        haze_image = cv2.imread(self.haze_path + '/' + haze_image_name)
        if self.transform1:
            haze_image = self.transform1(haze_image)
        if self.is_Gth:
            gt_image = cv2.imread(self.gt_path + '/' + haze_image_name[:-30] + '.bmp')
            if self.transform1:
                gt_image = self.transform1(gt_image)
        else:
            gt_image = False
        return haze_image, gt_image


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
