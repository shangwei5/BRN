import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
#from skimage import transform,data


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride, aug_times=1):   # R1400
    # train
    print('process training data')
    scales = [1]
    input_path = os.path.join(data_path, 'train', 'rainy_image')
    target_path = os.path.join(data_path, 'train', 'ground_truth')

    target_h5f = h5py.File('train_target.h5', 'w')
    input_h5f = h5py.File('train_input.h5', 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])
        #print(target.shape)
        h, w, c = target.shape

        for j in range(14):
            input_file = "%d_%d.jpg" % (i+1, j+1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

        #for k in range(len(scales)):
            target_img = target
            #target_img = np.expand_dims(target_img.copy(), 0)
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            #input_img = np.expand_dims(input_img.copy(), 0)
            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (
            input_file, target_patches.shape[3] * aug_times))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1
                # for m in range(aug_times-1):
                #    target_data_aug = data_augmentation(target_data, np.random.randint(1,8))
                #    target_h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=target_data_aug)
                #    train_num += 1
    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)
    '''   # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    '''


# print('val set, # samples %d\n' % val_num)


def prepare_data_heavyrain(data_path, patch_size, stride, aug_times=1):   #R100H
    # train
    print('process training data')
    scales = [1]
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)
    rain_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')
    save_rain_path = os.path.join(data_path, 'train_rain.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    rain_h5f = h5py.File(save_rain_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            #print(target.shape)
            h, w, c = target.shape

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                rain_file = "rainstreak-%d.png" % (i + 1)

                if os.path.exists(os.path.join(rain_path, rain_file)):
                    rain_img = cv2.imread(os.path.join(rain_path, rain_file))
                    b, g, r = cv2.split(rain_img)
                    rain_img = cv2.merge([r, g, b])

                    #for k in range(len(scales)):
                    target_img = target
                    #target_img = np.expand_dims(target_img.copy(), 0)
                    target_img = np.float32(normalize(target_img))
                    target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                    #input_img = np.expand_dims(input_img.copy(), 0)
                    input_img = np.float32(normalize(input_img))
                    input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                    # rain_img = np.expand_dims(rain_img.copy(), 0)
                    rain_img = np.float32(normalize(rain_img))
                    rain_patches = Im2Patch(rain_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                    print("target file: %s # samples: %d" % (
                     input_file, target_patches.shape[3] * aug_times))
                    for n in range(target_patches.shape[3]):
                        target_data = target_patches[:, :, :, n].copy()
                        target_h5f.create_dataset(str(train_num), data=target_data)

                        input_data = input_patches[:, :, :, n].copy()
                        input_h5f.create_dataset(str(train_num), data=input_data)

                        rain_data = rain_patches[:, :, :, n].copy()
                        rain_h5f.create_dataset(str(train_num), data=rain_data)
                        train_num += 1
                # for m in range(aug_times-1):
                #    target_data_aug = data_augmentation(target_data, np.random.randint(1,8))
                #    target_h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=target_data_aug)
                #    train_num += 1
    target_h5f.close()
    input_h5f.close()
    rain_h5f.close()
    print('training set, # samples %d\n' % train_num)
    '''   # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    '''




def prepare_data_lightrain1(data_path, patch_size, stride, aug_times=1):    #R100L with Data augmentation
    # train
    print('process training data')
    scales = [1]
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)
    rain_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'superincretrain_target.h5')
    save_input_path = os.path.join(data_path, 'superincretrain_input.h5')
    save_rain_path = os.path.join(data_path, 'superincretrain_rain.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    rain_h5f = h5py.File(save_rain_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])
        #print(target.shape)
        h, w, c = target.shape

        for j in range(3):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            rain_file = "rainstreak-%d.png" % (i + 1)
            rain_img = cv2.imread(os.path.join(rain_path, rain_file))
            b, g, r = cv2.split(rain_img)
            rain_img = cv2.merge([r, g, b])

        #for k in range(len(scales)):
            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)
                rain_img = cv2.flip(rain_img, 1)
            if j == 2:
                target_img = cv2.resize(target_img,(int(w * 1.2), int(h * 1.2)), interpolation=cv2.INTER_CUBIC)
                input_img = cv2.resize(input_img,(int(w * 1.2), int(h * 1.2)), interpolation=cv2.INTER_CUBIC)
                rain_img = cv2.resize(rain_img,(int(w * 1.2), int(h * 1.2)), interpolation=cv2.INTER_CUBIC)

            #target_img = np.expand_dims(target_img.copy(), 0)
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            #input_img = np.expand_dims(input_img.copy(), 0)
            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            # rain_img = np.expand_dims(rain_img.copy(), 0)
            rain_img = np.float32(normalize(rain_img))
            rain_patches = Im2Patch(rain_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (
            input_file, target_patches.shape[3] * aug_times))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                rain_data = rain_patches[:, :, :, n].copy()
                rain_h5f.create_dataset(str(train_num), data=rain_data)
                train_num += 1
                # for m in range(aug_times-1):
                #    target_data_aug = data_augmentation(target_data, np.random.randint(1,8))
                #    target_h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=target_data_aug)
                #    train_num += 1
    target_h5f.close()
    input_h5f.close()
    rain_h5f.close()
    print('training set, # samples %d\n' % train_num)
    '''   # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    '''

# print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):   #R100H
    def __init__(self, train=True, data_path='.'):
        super(Dataset, self).__init__()
        self.train = train
        self.data_path = data_path
        if self.train:
            target_path = os.path.join(self.data_path, 'train_target.h5')
            input_path = os.path.join(self.data_path, 'train_input.h5')
            rain_path = os.path.join(self.data_path, 'train_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')
            rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        rain_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            target_path = os.path.join(self.data_path, 'train_target.h5')
            input_path = os.path.join(self.data_path, 'train_input.h5')
            rain_path = os.path.join(self.data_path, 'train_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')
            rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        rain = np.array(rain_h5f[key])
        target_h5f.close()
        input_h5f.close()
        rain_h5f.close()
        return torch.Tensor(input), torch.Tensor(target), torch.Tensor(rain)



class Dataset1(udata.Dataset):    #R1400
    def __init__(self, train=True, data_path='.'):
        super(Dataset1, self).__init__()
        self.train = train
        self.data_path = data_path
        if self.train:
            target_path = os.path.join(self.data_path, 'train', 'ground_truth', 'train_target.h5')
            input_path = os.path.join(self.data_path, 'train', 'rainy_image', 'train_input.h5')
            #rain_path = os.path.join(self.data_path, 'train_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path,'r')
            #rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        #rain_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            target_path = os.path.join(self.data_path, 'train', 'ground_truth', 'train_target.h5')
            input_path = os.path.join(self.data_path, 'train', 'rainy_image', 'train_input.h5')
            #rain_path = os.path.join(self.data_path, 'train_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')
            #rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        #rain = np.array(rain_h5f[key])
        target_h5f.close()
        input_h5f.close()
        #rain_h5f.close()
        return torch.Tensor(input), torch.Tensor(target)#, torch.Tensor(rain)




class Dataset3(udata.Dataset):     #R100L with Data augmentation
    def __init__(self, train=True, data_path='.'):
        super(Dataset3, self).__init__()
        self.train = train
        self.data_path = data_path
        if self.train:
            target_path = os.path.join(self.data_path, 'superincretrain_target.h5')
            input_path = os.path.join(self.data_path, 'superincretrain_input.h5')
            rain_path = os.path.join(self.data_path, 'superincretrain_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')
            rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        rain_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            target_path = os.path.join(self.data_path, 'superincretrain_target.h5')
            input_path = os.path.join(self.data_path, 'superincretrain_input.h5')
            rain_path = os.path.join(self.data_path, 'superincretrain_rain.h5')
            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')
            rain_h5f = h5py.File(rain_path, 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        rain = np.array(rain_h5f[key])
        target_h5f.close()
        input_h5f.close()
        rain_h5f.close()
        return torch.Tensor(input), torch.Tensor(target), torch.Tensor(rain)


