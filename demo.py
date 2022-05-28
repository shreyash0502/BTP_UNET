import argparse
import cv2
import numpy as np
#from resnet import Bottleneck as ResBlock
#from sharpnet_model import *
from PIL import Image
import os, sys
from imageio import imread, imwrite
import scipy.io as sio
import loaddata_demo as loaddata
import matplotlib.image
import matplotlib.pyplot as plt
from SUMNet import *

#plt.set_cmap("jet")
'''
def round_down(num, divisor):
    return num - (num % divisor)

def get_pred_from_input(image_pil, args):
    #seg = None
    #boundary = None
    depth = None

    image_np = np.array(image_pil)
    w, h = image_pil.size

    scale = args.rescale_factor

    h_new = round_down(int(h * scale), 16)
    w_new = round_down(int(w * scale), 16)

    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
        print("Input image has only 1 channel, please use an RGB or RGBA image")
        sys.exit(0)

    if len(image_np.shape) == 4 or image_np.shape[-1] == 4:
        # RGBA image to be converted to RGB
        image_pil = image_pil.convert('RGBA')
        image = Image.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        image.paste(image_pil.copy(), mask=image_pil.split()[3])
    else:
        image = image_pil

    image = image.resize((w_new, h_new), Image.ANTIALIAS)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = []
    t.extend([ToTensor(), normalize])
    transf = Compose(t)

    data = [image, None]
    image = transf(*data)

    image = torch.autograd.Variable(image).unsqueeze(0)
    image = image.to(device)

    depth_pred = model(image)
    tmp = depth_pred.data.cpu()
    shp = tmp.shape[2:]

    mask_pred = np.ones(shape=shp)
    mask_display = mask_pred


    depth_pred = depth_pred.data.cpu().numpy()[0, 0, ...] * 65535 / 1000
    depth_pred = (1 / scale) * cv2.resize(depth_pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    image_path = args.image_path
    outpath =  args.outpath
    img_name = os.path.basename(image_path).rsplit(".")[0]
    sio.savemat( os.path.join(outpath,img_name +'_depth.mat'), {'depth': depth_pred})
    m = np.min(depth_pred)
    M = np.max(depth_pred)
    depth_pred = (depth_pred - m) / (M - m)
    depth = Image.fromarray(np.uint8(plt.cm.jet(depth_pred) * 255))
    depth = np.array(depth)#[:, :, :3]

    return tuple([depth])


def save_preds(outpath, preds, img_name):
    suffixes = ['_depth.png', '_img.png']
    for k, pred in enumerate(preds):
        if pred is not None:
            imwrite(os.path.join(outpath, img_name + suffixes[k]), pred)

'''
image_path = '6.jpg'

net = SUMNet()
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('training on GPU')
    net = net.cuda()


torch.set_grad_enabled(False)

net_dict = net.state_dict()

# Load model
trained_model_path = './checkpoint_0_iter_12000.pth'
# trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)

# # load image resnet encoder and mask_encoder and normals_decoder (not depth_decoder or normal resnet)
# model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}

# model.load_state_dict(model_weights)
net.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage))
net.eval()
#model.to(device)


nyu2_loader = loaddata.readNyu2(image_path)

for i, image in enumerate(nyu2_loader):
    image = torch.autograd.Variable(image, volatile=True)#.cuda()
    # image = image.data.numpy()
    # print('shape', np.shape(image))
    # print(np.min(image[0]), np.max(image[0]))

    # image = np.transpose(image[0], (1,2,0))
    # Image.fromarray(np.uint8(image* 255), mode ='RGB').save('./tmp/img_{}.png'.format(i))

    depth_pred  = net(image)
    # print(out.view(out.size(2),out.size(3)).data.cpu().numpy().shape)
    # out = out.data.cpu().numpy()[0, 0, ...]#/10.0
    # m = np.min(out)
    # M = np.max(out)
    # print(m, M)
    # out= (out - m) / (M - m)
    # # # #print(out.shape)
    # out = np.uint8((out) * 255) 
    # print(out.shape)
    # # #out = Image.fromarray((out))
    # # #out = np.array(out)[:, :, :3]
    
    # #matplotlib.image.imsave('tmp/out.png', out)
    # matplotlib.image.imsave('tmp/out.png', out)

    r = depth_pred.data.numpy()[0, 0, ...]
    print(r.shape)
    #m = np.min(depth_pred)
    #M = np.max(depth_pred)
    # print(m, M)
    # print(depth_pred.shape)
    #depth_pred =  cv2.resize(depth_pred, dsize=(304, 228), interpolation=cv2.INTER_LINEAR)
    #m = np.min(depth_pred)
    #M = np.max(depth_pred)
    #depth_pred = (depth_pred - m) / (M - m)
	torchvision.utils.save_image( r[0,:,:,:], './tmp/pred_dep_{}.png'.format(6), normalize=True )
    
    #depth = Image.fromarray(np.uint8(plt.cm.jet(depth_pred) * 255), mode='RGB').save('./tmp/out_g.jpg')
    # depth = np.array(depth)#[:, :, :3]
    # plt.imshow(depth)
    # plt.show()
    
