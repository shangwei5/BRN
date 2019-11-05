import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from models import DnCNN
from utils import *

from generator import Generator_prelstminter22, print_network
import time

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


parser = argparse.ArgumentParser(description="BRN_Test")
parser.add_argument("--logdir", type=str, default="logs/BRN/R100H", help='path of log files')
parser.add_argument("--data_path", type=str, default="dataset/...", help='path to training data')
parser.add_argument("--save_path", type=str, default="result/BRN/R100H/output", help='path to save results')
parser.add_argument("--save_path_r", type=str, default="result/BRN/R100H/rainstreak", help='path to save rain streaks')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=4, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


# opt.save_path = os.path.join(opt.data_path, opt.save_path)
# if not os.path.exists(opt.save_path):
#     os.mkdir(opt.save_path)

def normalize(data):
    return data/255.

def main():
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.isdir(opt.save_path_r):
        os.makedirs(opt.save_path_r)
    # Build model
    print('Loading model ...\n')

    model = Generator_prelstminter22(opt.inter_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()

    # if model is trained by multiGPU

    # state_dict = torch.load(os.path.join(opt.logdir, 'net_latest.pth'))
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    # if model is trained by single GPU
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.data_path, 'rainy/*.png'))

    files_source.sort()
    # process data
    time_test = 0
    i = 1
    for f in files_source:
        img_name = os.path.basename(f)

        # image
        Img = cv2.imread(f)
        h, w, c = Img.shape

        b, g, r = cv2.split(Img)
        Img = cv2.merge([r, g, b])
        #Img = cv2.resize(Img, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
        '''
        if h > 1024:
            ratio = 1024.0/h
            Img = cv2.resize(Img,(int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)

        if w > 1024:
            ratio = 1024.0/w
            Img = cv2.resize(Img,(int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)  #4x4像素邻域的双三次插值
        '''
        Img = normalize(np.float32(Img))
        Img = np.expand_dims(Img.transpose(2, 0, 1), 0)
        #Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        #noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource #+ noise

        if opt.use_GPU:
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        else:
            ISource, INoisy = Variable(ISource), Variable(INoisy)

        with torch.no_grad(): # this can save much memory
            torch.cuda.synchronize()
            start_time = time.time()
            out, _, out_r, _ = model(INoisy)
            out = torch.clamp(out, 0., 1.)
            out_r = torch.clamp(out_r, 0., 1.)

            torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            print(img_name)
            print(dur_time)
            time_test += dur_time
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        #psnr = batch_PSNR(Out, ISource, 1.)
        #psnr_test += psnr
        #print("%s PSNR %f" % (f, psnr))
        if opt.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #计算之后放回cpu储存
            save_out_r = np.uint8(255 * out_r.data.cpu().numpy().squeeze())

        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())
            save_out_r = np.uint8(255 * out_r.data.numpy().squeeze())

        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])
        # cv2.imshow('a',save_out)

        save_out_r = save_out_r.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out_r)
        save_out_r = cv2.merge([r, g, b])


        save_path = opt.save_path
        save_path_r = opt.save_path_r

        cv2.imwrite(os.path.join(save_path, img_name), save_out)
        cv2.imwrite(os.path.join(save_path_r, img_name), save_out_r)

        i = i + 1

    print(time_test/i)

    #psnr_test /= len(files_source)
    #print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

