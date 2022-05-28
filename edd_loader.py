import os
import sys
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import random 
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from augmentation import *
class eddLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
        self,
        root,
        sbd_path=None,
        split="train",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        
        # self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 1
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if not self.test_mode:
            for split in ["train", "test"]:
        train_path = pjoin(self.root, "nyu2_train.csv") # /content/data/nyu2_train.csv
        val_path = pjoin(self.root, "nyu2_test.csv")
    
        self.files['train'] = pd.read_csv(train_path)
        self.files['val'] = pd.read_csv(val_path)
        print(len(self.files['train']))
        print(len(self.files['train']))
        self.files[self.split]
        sys.exit(0)
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index].split(".")[0]
        im_path = pjoin(self.root, "originalImages", im_name + ".jpg")
        BE_lbl_path = pjoin(self.root, "masks", im_name + "_BE.tif")
        cancer_lbl_path = pjoin(self.root, "masks", im_name + "_cancer.tif")
        HGD_lbl_path = pjoin(self.root, "masks", im_name + "_HGD.tif")
        polyp_lbl_path = pjoin(self.root, "masks", im_name + "_polyp.tif")
        suspicious_lbl_path = pjoin(self.root, "masks", im_name + "_suspicious.tif")
        # print(len(os.listdir(BE_lbl_path)))


        im = Image.open(im_path)
        w, h = 260,260
        #im = im.resize((self.img_size[0], self.img_size[1]))
        #lbl = np.zeros([self.img_size[0], self.img_size[1],5], dtype = np.uint8)

        if os.path.exists(BE_lbl_path):
            BE_lbl = Image.open(BE_lbl_path).resize((w, h))
            #lbl[:, :,0] = BE_lbl
        else:
            BE_lbl = Image.fromarray(np.zeros([w, h], dtype = np.uint8), mode="L")


        if os.path.exists(suspicious_lbl_path):
            suspicious_lbl = Image.open(suspicious_lbl_path).resize((w, h))
            #lbl[:, :, 1] = suspicious_lbl
        else:
            suspicious_lbl = Image.fromarray(np.zeros([w, h], dtype = np.uint8), mode="L")

        if os.path.exists(HGD_lbl_path):
            HGD_lbl= Image.open(HGD_lbl_path).resize((w, h))
            #lbl[:,:,2] = HGD_lbl
        else:
            HGD_lbl = Image.fromarray(np.zeros([w, h], dtype = np.uint8), mode="L")

        if os.path.exists(cancer_lbl_path):
            cancer_lbl = Image.open(cancer_lbl_path).resize((w, h))
            #lbl[:,:,3] = cancer_lbl
        else:
            cancer_lbl = Image.fromarray(np.zeros([w, h], dtype = np.uint8), mode="L")

        if os.path.exists(polyp_lbl_path):
            polyp_lbl = Image.open(polyp_lbl_path).resize((w, h))
            #lbl[:,:,4] = polyp_lbl
        else:
            polyp_lbl = Image.fromarray(np.zeros([w, h], dtype = np.uint8), mode="L")

        #lbl = lbl.transpose([2,0,1])


        if self.augmentations is not None:
            # print('Image:',type(im))
            # print('Label:',type(lbl))
            im, BE_lbl,suspicious_lbl,HGD_lbl,cancer_lbl,polyp_lbl = self.augmentations(im, BE_lbl,suspicious_lbl,HGD_lbl,cancer_lbl,polyp_lbl)
            # print(lbl.shape)
            # lbl[0,:,:] = BE_lbl
            # lbl[1,:,:] = suspicious_lbl
            # lbl[2,:,:] = HGD_lbl
            # lbl[3,:,:] = cancer_lbl
            # lbl[4,:,:] = polyp_lbl
        if self.is_transform:
            im, lbl = self.transform(im, BE_lbl,suspicious_lbl,HGD_lbl,cancer_lbl,polyp_lbl)

        return im, lbl

    def transform(self, img, BE_lbl,suspicious_lbl,HGD_lbl,cancer_lbl,polyp_lbl ):
        #if self.img_size == ("same", "same"):
        #    pass
        #else:
        img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        BE_lbl = BE_lbl.resize((self.img_size[0], self.img_size[1]))
        suspicious_lbl = suspicious_lbl.resize((self.img_size[0], self.img_size[1]))
        HGD_lbl = HGD_lbl.resize((self.img_size[0], self.img_size[1]))
        cancer_lbl =cancer_lbl.resize((self.img_size[0], self.img_size[1])) 
        polyp_lbl = polyp_lbl.resize((self.img_size[0], self.img_size[1]))

        lbl = np.zeros([self.img_size[0], self.img_size[1],5], dtype = np.uint8)
        lbl = lbl.transpose([2,0,1])
        lbl[0,:,:] = BE_lbl
        lbl[1,:,:] = suspicious_lbl
        lbl[2,:,:] = HGD_lbl
        lbl[3,:,:] = cancer_lbl
        lbl[4,:,:] = polyp_lbl

        
        #    lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).float()
        #lbl[lbl == 255] = 0
        return img, lbl






'''
# Leave code for debugging purposes
import augmentation as aug
if __name__ == '__main__':
    local_path = '/home/histosr/Desktop/Media/EDD2020_release-I_2020-01-15_v2_s3/EDD2020_release-I_2020-01-15/'
    bs = 1
    augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5)])
    dst = eddLoader(root=local_path, is_transform=True, augmentations=augs,split='train')
    # print(len(dst))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(labels.dtype)
        break
# imgs = imgs.numpy()[:, ::-1, :, :]
# imgs = np.transpose(imgs, [0,2,3,1])
# f, axarr = plt.subplots(bs, 2)
# for j in range(bs):
# axarr[j][0].imshow(imgs[j])
# axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
# plt.show()
# a = raw_input()
# if a == 'ex':
# break
# else:
# plt.close()
'''