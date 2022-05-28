#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:27:04 2018

@author: sumanthnandamuri
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import tqdm
import time
import gc


from torch.utils import data

# from aug_utils import *
import loaddata
import sys
import os
from PIL import Image
from SUMNet import *
from utils import *

data_path = 'D:/Downloads/CODE/CODE/data/'
# data_path = '/home/vajir/Desktop/MedIA/EDD2020_release-I_2020-01-15_v2_s3/EDD2020_release-I_2020-01-15/'
# augmentations = {'gamma': 1.5,
#                 'hue': 0.5, 
#                 'brightness': 0.5, 
#                 'saturation': 0.5, 
#                 'contrast': 0.5}

# data_aug = get_composed_augmentations(augmentations)
# data_aug = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5),aug.RandomHorizontallyFlip(0.5),\
#     aug.AdjustGamma(1.5),aug.AdjustSaturation(0.5),aug.AdjustHue(0.5),aug.AdjustBrightness(0.5),aug.AdjustContrast(0.5),\
#     aug.RandomCrop(256,256)])
# data_aug = aug.Compose([aug.AdjustGamma(1.5),aug.AdjustSaturation(0.5),aug.AdjustHue(0.5),aug.AdjustBrightness(0.5),\
    # aug.AdjustContrast(0.5), aug.RandomHorizontallyFlip(0.5),aug.RandomHorizontallyFlip(0.5)])
# data_aug = None

# savePath = 'Results/unet_with_augs/'
# if not os.path.isdir(savePath):
#     os.makedirs(savePath)

BatchSize = 4

trainDataLoader = loaddata.getTrainingData(batch_size = BatchSize)


validDataLoader = loaddata.getTestingData(batch_size = BatchSize)

net = SUMNet()
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('training on GPU')
    net = net.cuda()
#print(netS)
#sys.exit(0) 
learning_rate = 1e-4  
#optimizer = optim.Adam(net.parameters(), lr = 1e-3)
optimizer  = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))#

# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([2,4,5,7,3]).cuda)
criterion = None

def train(trainDataLoader, validDataLoader, net, optimizer, criterion, use_gpu):
    epochs = 3
    trainLoss = []
    validLoss = []
    trainDiceCoeff = []
    validDiceCoeff = []
    start = time.time()
    
    #keep the track of total epoch to cange the learning rate
    total_epoch = 0
    max_epoch = (epochs*50000)/BatchSize
    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        # trainDice = np.zeros(5)
        # validDice = np.zeros(5)
        
        net.train(True)
        for i, sample_batched in enumerate(trainDataLoader):
            inputs, labels = sample_batched['image'], sample_batched['depth']
            # check here the values of the inputs and the labels
            # labels = np.expand_dims(labels, axis =1)

            # inputs = inputs.data.numpy()
            # labels = labels.data.numpy()
            # print(inputs.shape)
            # print(labels.shape)
            # print('min and max input', np.min(inputs[0]), np.max(inputs[0]))
            # print('min ans max depth', np.min(labels[0]), np.max(labels[0]))
            # for j in range(4):
                    
            #     r = np.transpose(inputs[j], (1,2,0))
            #     d = np.transpose(labels[j], (1,2,0))[:, :, 0]#/10
            #     print('max value min rgb',np.max(r), np.min(r))
            #     print('min val max d',np.max(d), np.min(d))
            #     r =  Image.fromarray(np.uint8(r * 255), mode ='RGB').save('./tmp/img_{}.png'.format(j))
            #     d = Image.fromarray(np.uint8(d * 255), mode='L').save('./tmp/dep_{}.png'.format(j))
            #     #imwrite('img_{}.jpg'.format(j), r)
            #     #imwrite('dep_{}.jpg'.format(j), d)
            # sys.exit(0)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            probs = net(inputs)

            loss = (torch.abs(probs -labels)).mean()
            if trainBatches % 20 == 0:
                print(' epoch : {}  | iter : {}  | loss :  {:.3f} '.format(epoch, trainBatches+1, loss.item()))
                trainLoss.append([loss.item(), (epoch+1)*i])

            # change the learning rate

            total_epoch += 1
            if total_epoch % 100 == 0:
                adjust_learning_rate(learning_rate, 'poly',  max_epoch,optimizer, total_epoch)

  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainRunningLoss += loss.item()

            


            trainBatches += 1
            if trainBatches % 100 == 0:
            	print('saving the train imagesS')
            	torch.save(
                	net.state_dict(),

                	"checkpoint_{}_iter_{}.pth".format(epoch, trainBatches),)
            	#save some of the images , gt and pred
            	inputs = inputs.data.cpu().numpy()
            	labels = labels.data.cpu().numpy()
            	probs = probs.data.cpu().numpy()
            	for j in range(BatchSize):
            		r = np.transpose(inputs[j], (1,2,0))
            		l = np.transpose(labels[j], (0,1))#[:, :, 0]
            		p = np.transpose(probs[j], (1,2,0))[:, :, 0]
            		Image.fromarray(np.uint8(r * 255), mode ='RGB').save('./tmp/img_{}_{}_{}.png'.format(epoch,trainBatches, j))
            		Image.fromarray(np.uint8(l * 255), mode='L').save('./tmp/depGT_{}_{}_{}.png'.format(epoch,trainBatches, j))
            		Image.fromarray(np.uint8(p * 255), mode='L').save('./tmp/depPred_{}_{}_{}.png'.format(epoch,trainBatches, j))
       

            # if trainBatches ==1:
            # 	break
            
        # trainLoss.append(trainRunningLoss/trainBatches)




        
        # net.train(False)
        # for i, sample_batched in enumerate(validDataLoader):
        #     inputs, labels = sample_batched['image'], sample_batched['depth']
    

        #     if use_gpu:
        #         inputs = inputs.cuda()
        #         labels = labels.cuda()
        #     probs = net(inputs)  

        #     loss = torch.abs(probs -labels).mean()


        #     # validDice += dice_coefficient(preds, labels).item()
        #     # for classNum in range(5):
        #     #     validDice[classNum] += dice_coefficient(preds[:,classNum],labels[:,classNum])[1]


        #     validRunningLoss += loss.item()
        #     validBatches += 1
        #     if validBatches == 4:
        #         break
        # validLoss.append(validRunningLoss/validBatches)
        # validDiceCoeff.append(validDice/validBatches)
        # if validDice[0] > bestValidDice[0]:
        #     bestValidDice[0] = validDice[0]
        #     torch.save(net.state_dict(), savePath+'SUMNet_class0_best.pt')
        # if validDice[1] > bestValidDice[1]:
        #     bestValidDice[1] = validDice[1]
        #     torch.save(net.state_dict(), savePath+'SUMNet_class1_best.pt')
        # if validDice[2] > bestValidDice[2]:
        #     bestValidDice[2] = validDice[2]
        #     torch.save(net.state_dict(), savePath+'SUMNet_class2_best.pt')
        # if validDice[3] > bestValidDice[3]:
        #     bestValidDice[3] = validDice[3]
        #     torch.save(net.state_dict(), savePath+'SUMNet_class3_best.pt')
        # if validDice[4] > bestValidDice[4]:
        #     bestValidDice[4] = validDice[4]
        #     torch.save(net.state_dict(), savePath+'SUMNet_class4_best.pt')

        epochEnd = time.time()-epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f}  |'\
              .format(epoch+1, epochs, trainRunningLoss/trainBatches))
        # print('Train Dice : {:.3f},{:.3f},{:.3f},{:.3f},{:.3f} '.format(trainDice[0]/trainBatches,trainDice[1]/trainBatches,trainDice[2]/trainBatches,trainDice[3]/trainBatches,trainDice[4]/trainBatches))
        # print('Valid Dice : {:.3f},{:.3f},{:.3f},{:.3f},{:.3f} '.format(validDice[0]/validBatches,validDice[1]/validBatches,validDice[2]/validBatches,validDice[3]/validBatches,validDice[4]/validBatches))
        print('Time: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
        # break
    end = time.time()-start
    print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))

    # torch.save(trainLoss,savePath+'trainLoss.pt')
    # torch.save(validLoss,savePath+'validLoss.pt')
    # torch.save(trainDiceCoeff,savePath+'trainDiceCoeff.pt')
    # torch.save(validDiceCoeff,savePath+'validDiceCoeff.pt')

    # trainLoss = trainLoss[:]
    # trainLoss = np.array(trainLoss)
    # validLoss = np.array(validLoss)

    DF = pd.DataFrame({'Train Loss': trainLoss})
    # final checkpoints
    
    torch.save(net.state_dict(),
                "final_checkpoint_{}.pth".format(epoch),)
    return DF


DF = train(trainDataLoader, validDataLoader, net, optimizer, criterion, use_gpu)
DF.to_csv('SUMNet.csv')



