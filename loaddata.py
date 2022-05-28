import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import sys
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import h5py


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        path_name = self.frame.iloc[idx, 0]
        # depth_name = self.frame.iloc[idx, 1]
        # image = Image.open(image_name)
        # depth = Image.open(depth_name)

        file_name = h5py.File(path_name, 'r')

        rgb_h5 = file_name['rgb'][:].transpose(1, 2, 0)
        dep_h5 = file_name['depth'][:]

        image = Image.fromarray(rgb_h5, mode='RGB')
        depth = Image.fromarray(dep_h5.astype('float32'), mode='F')

        # rgb  = image.getextrema()
        # dep = depth.getextrema()
        # print('*'*10)
        # print(rgb)
        # print(dep)
        # print('*'*10)
        # for debug purpose
        # print('before trannsform')
        # print(image.size, depth.size)
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=4):
    # __imagenet_pca = {
    #     'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    #     'eigvec': torch.Tensor([
    #         [-0.5675,  0.7192,  0.4009],
    #         [-0.5808, -0.0045, -0.8140],
    #         [-0.5836, -0.6948,  0.4203],
    #     ])
    # }
    # __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
    #                     'std': [0.229, 0.224, 0.225]}

    # transformed_training = depthDataset(csv_file='./data/nyu2_train.csv',
    #                                     transform=transforms.Compose([
    #                                         # Scale(240),
    #                                         RandomHorizontalFlip(),
    #                                         RandomRotate(5),
    #                                         CenterCrop([304, 228], [304, 228]),#
    #                                         ToTensor()#,
    #                                         #Lighting(0.1, __imagenet_pca[
    #                                         #    'eigval'], __imagenet_pca['eigvec']),
    #                                         # ColorJitter(
    #                                         #     brightness=0.4,
    #                                         #     contrast=0.4,
    #                                         #     saturation=0.4,
    #                                         # ),
    #                                         # Normalize(__imagenet_stats['mean'],
    #                                         #            __imagenet_stats['std'])
    #                                     ]))
    transformed_training = depthDataset(csv_file='./data/nyu2_train.csv',
                                        transform=transforms.Compose([
                                            ToTensor(),
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=0, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           # Scale(240),
                                           # CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=False),
                                           # Normalize(__imagenet_stats['mean'],
                                           #           __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing

# FOR DEBUGGING PURPOSES
# sample = getTrainingData(batch_size=4)

# for i, sample_batched in enumerate(sample):
#     img, dep =  sample_batched['image'], sample_batched['depth']

#     #save the image 
#     img = img.data.numpy()
#     dep = dep.data.numpy()
#     # expanding the dim of depth in channels
#     dep =  np.expand_dims(dep, axis =1)
#     print('rgb', np.shape(img))
#     print('dep', np.shape(dep))

#     for j in range(4):

#         r = np.transpose(img[j], (1,2,0))
#         d = np.transpose(dep[j], (1,2,0))[:, :, 0]#/10
#         # print('max value',np.max(r))
#         # print('min val',np.max(d))

#         r =  Image.fromarray(np.uint8(r * 255), mode ='RGB').save('./tmp/img_{}.png'.format(j))
#         d = Image.fromarray(np.uint8(d * 255), mode='L').save('./tmp/dep_{}.png'.format(j))
#         #imwrite('img_{}.jpg'.format(j), r)
#         #imwrite('dep_{}.jpg'.format(j), d)
#     break
