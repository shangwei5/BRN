import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from heavy_DeblurDataset import prepare_data_lightrain1, Dataset3
from utils import *
import re

import cv2
from scipy import signal
from scipy import misc
from torch.optim.lr_scheduler import MultiStepLR

import pytorch_ssim

from generator import Generator_prelstminter22, print_network

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Recurrent")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs/BRN/R100L", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="/home/shangwei/dataset/RainTrainH_wo_test",help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=4, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id




def main():
    if not os.path.isdir(opt.outf):
        os.makedirs(opt.outf)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset3(train=True, data_path=opt.data_path)
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # vgg = Vgg16(requires_grad=False).cuda()
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    # net.apply(weights_init_kaiming)

    net = Generator_prelstminter22(recurrent_iter=opt.inter_iter, use_GPU=opt.use_GPU)
    #net = nn.DataParallel(net)
    print_network(net)

    #criterion = nn.MSELoss(size_average=False)
    criterion = pytorch_ssim.SSIM()

    # Move to GPU

    model = net.cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)  # learning rates
    #scheduler = MultiStepLR(optimizer, milestones=[120, 140], gamma=0.2)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'

    initial_epoch = findLastCheckpoint(save_dir=opt.outf)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_epoch%d.pth' % initial_epoch)))

    for epoch in range(initial_epoch, opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        scheduler.step(epoch)
        # set learning rate
        for param_group in optimizer.param_groups:
            #param_group["lr"] = current_lr
            print('learning rate %f' % param_group["lr"])
        # train
        for i, (input, target, rain) in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            #rain = input - target
            input_train, target_train, rain_train = Variable(input.cuda()), Variable(target.cuda()),Variable(rain.cuda())

            out_train, outs, out_r_train, outs_r = model(input_train)



#            tv_loss = tv(out_train)
            pixel_loss = criterion(target_train, out_train)
            #pixel_loss_r = criterion(rain_train, out_r_train)

            # loss1 = criterion(target_train, outs[0])
            # loss2 = criterion(target_train, outs[1])
            # loss3 = criterion(target_train, outs[2])
            # loss4 = criterion(target_train, outs[3])
            #
            # #
            # loss1_r = criterion(rain_train, outs_r[0])
            # loss2_r = criterion(rain_train, outs_r[1])
            # loss3_r = criterion(rain_train, outs_r[2])
            # loss4_r = criterion(rain_train, outs_r[3])

            '''
            y = utils_vgg.normalize_batch(out_train)
            x = utils_vgg.normalize_batch(target_train)
            x, y = target_train, out_train

            features_y = vgg(y)
            features_x = vgg(x)

            perceptral_loss = 0e-5 * criterion(features_y.relu2_2, features_x.relu2_2)
            '''

            # loss = -(pixel_loss + 0.5 * (loss1 + loss2 + loss3 + loss4)) #/(target_train.size()[0] * 2) + loss5 + loss6 + loss7 + loss8
            # loss_r = -(pixel_loss_r + 0.5 * (loss1_r + loss2_r + loss3_r + loss4_r)) #+ loss5_r + loss6_r + loss7_r + loss8_r
            # lossm = 0.55 * loss + 0.45 * loss_r
            loss = - (pixel_loss)
            loss.backward()
            # loss.backward(retain_graph=True)
            # loss_r.backward()
            optimizer.step()
            # results
            model.eval()
            out_train, _, out_r_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            out_r_train = torch.clamp(out_r_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            psnr_train_r = batch_PSNR(out_r_train, rain_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                #writer.add_scalar('loss_r', loss_r.item(), step)
                writer.add_scalar('PSNR_r on training data', psnr_train_r, step)
            step += 1
        ## the end of each epoch

        model.eval()
        '''
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        '''
        # log the images
        out_train, _, out_r_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        out_r_train = torch.clamp(out_r_train, 0., 1.)
        Img = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        rainstreak = utils.make_grid(out_r_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        writer.add_image('estimated rain image', rainstreak, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_epoch%d.pth' % (epoch + 1)))




if __name__ == "__main__":
    if opt.preprocess:
        prepare_data_lightrain1(data_path=opt.data_path, patch_size=100, stride=100, aug_times=1)

    main()
