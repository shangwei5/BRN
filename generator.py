#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os


def print_network(net):
	num_params = 0
	for param in net.parameters():   #网络的参数都保存在parameters()函数当中
		num_params += param.numel()  #param.numel()返回元素个数
	print(net)
	print('Total number of parameters: %d' % num_params)




#Model: (lstm + 5 resnet + conv : recurrent_iter)rainstreak,rainstreak+input+x(rs+x+y)   ----CRN
class Generator_prelstm(nn.Module):
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_prelstm, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list






#Model: (lstm + 5 resnet + conv : recurrent_iter)   #BRN
class Generator_prelstminter22(nn.Module):
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_prelstminter22, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list


#Model: (lstm + 5 resnet + conv : recurrent_iter)
class Generator_prelstminter11(nn.Module):    #---BRN-RX
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_prelstminter11, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list


#Model: (lstm + 5 resnet + conv : recurrent_iter)
class Generator_prelstminter33(nn.Module):   #----BRN-XR
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_prelstminter33, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list




class Generator_preconvinter22(nn.Module):    #RRNDSI-conv
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_preconvinter22, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_h = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_h_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)

            h_r = self.conv_h_r(r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)

            h = self.conv_h(x)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list


class Generator_preresinter22(nn.Module):    #RRNDSI-Res
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(Generator_preresinter22, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_h = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_h_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)

            h_r = self.conv_h_r(r) + h_r
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv4_r(r) + resr)
            # resr = r
            # r = F.relu(self.res_conv5_r(r) + resr)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)

            h = self.conv_h(x) + h
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list, r, r_list