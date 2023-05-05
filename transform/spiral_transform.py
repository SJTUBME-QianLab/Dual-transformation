# encoding: utf-8

"""
Read images and corresponding labels.
三线性插值
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import cv2
import time
import sys

# dx=int(sys.argv[1])
# dy=int(sys.argv[2])
# dz=int(sys.argv[3])
# print(dx,dy,dz)
# dx, dy, dz = 0, 0, 0
R = 40
N = 20  # fai转了5圈
modal = 'a'


class DataSet():
    def __init__(self, image_file, mask_file, image_list_file, data_dir_new, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names, image_mask_names = [], []
        with open(image_list_file) as f:
            for line in f:
                items = line.split()
                image_name = os.path.join(image_file, items[0] + modal + '.npy')
                image_mask_name = os.path.join(mask_file, items[0] + modal + '.npy')
                # start = time.clock()
                # image1.append(nrrd.read(image_name[0])[0].transpose(1, 0, 2))
                # print("Time used:", (time.clock() - start))
                image_names.append(image_name)
                image_mask_names.append(image_mask_name)

                # print(i)

        self.image_names = image_names
        self.image_mask_names = image_mask_names
        # self.image1 = image1
        self.data_dir_new = data_dir_new
        self.transform = transform

    def data_new(self, index):  # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        print(self.image_names[index % len(self.image_names)])
        # image = Image.open(image_name).convert('RGB签
        if index < len(self.image_names) * 3:  # 000
            image1 = np.load(self.image_names[index % len(self.image_names)])
            image1_mask = np.load(self.image_mask_names[index % len(self.image_names)])

            if index < len(self.image_names):
                trans = '000000'  # xyz yzx zxy
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 2:
                trans = '000hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:  # 垂直
                trans = '000ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')

        elif index < len(self.image_names) * 6:  # l10
            image1 = image_trans(np.load(self.image_names[index % len(self.image_names)]), angle='l40')
            image1_mask = image_trans(np.load(self.image_mask_names[index % len(self.image_names)]), angle='l40')
            if index < len(self.image_names) * 4:
                trans = 'l40000'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 5:
                trans = 'l40hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'l40ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')
        else:
        # elif index < len(self.image_names) * 9:  # r10
            image1 = image_trans(np.load(self.image_names[index % len(self.image_names)]), angle='r40')
            image1_mask = image_trans(np.load(self.image_mask_names[index % len(self.image_names)]), angle='r40')
            if index < len(self.image_names) * 7:
                trans = 'r40'
                image1_trans = copy.deepcopy(image1)
                image1_mask_trans = copy.deepcopy(image1_mask)
            elif index < len(self.image_names) * 8:
                trans = 'r40hor'
                image1_trans = image_trans(image1, angle='flip_horizontally')
                image1_mask_trans = image_trans(image1_mask, angle='flip_horizontally')
            else:
                trans = 'r40ver'
                image1_trans = image_trans(image1, angle='flip_vertically')
                image1_mask_trans = image_trans(image1_mask, angle='flip_vertically')

        image1_slice = rotate_slice_evenly(image1_trans, self.image_names[index % len(self.image_names)],
                                           trans, image1_mask_trans, self.data_dir_new)


        return image1_slice

    def data_length(self):
        return len(self.image_names)


def setpara(image):
    zz, xx, yy = image.shape
    mid_z = int(zz / 2)
    mid_x = int(xx / 2)
    mid_y = int(yy / 2)

    return mid_z, mid_x, mid_y


def image_trans(im1, angle):
    image1 = copy.deepcopy(im1)
    mid_z1, mid_x1, mid_y1 = setpara(image1)
    if angle[0] == 'l':  # 顺时针
        angle_num = int(angle[1:])
        rot_mat = cv2.getRotationMatrix2D((mid_y1, mid_x1), -angle_num, 1)
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.warpAffine(image1[i1, :, :], rot_mat,
                                              (image1[i1, :, :].shape[1], image1[i1, :, :].shape[0]))
    elif angle[0] == 'r':  # 逆时针
        angle_num = int(angle[1:])
        rot_mat = cv2.getRotationMatrix2D((mid_y1, mid_x1), angle_num, 1)
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.warpAffine(image1[i1, :, :], rot_mat,
                                              (image1[i1, :, :].shape[1], image1[i1, :, :].shape[0]))
    elif angle == 'flip_horizontally':  # 水平翻转
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.flip(image1[i1, :, :], 1)
    elif angle == 'flip_vertically':  # 垂直翻转
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.flip(image1[i1, :, :], 0)
    else:
        print('please choose right angle')

    return image1


def rotate_slice_evenly(nrrd_data, nrrd_filename, trans, image_mask, DATA_DIR_NEW):
    xx, yy, zz = image_mask.nonzero()
    mid_x = int(round((max(xx) + min(xx)) / 2))
    mid_y = int(round((max(yy) + min(yy)) / 2))
    mid_z = int(round((max(zz) + min(zz)) / 2))
    # 以mid_x, mid_y, mid_z为坐标原点建立坐标系

    k = int(2 * N * N / np.pi)  # 4 * N * N / np.pi
    rotate_slice = np.zeros((2 * R, k))
    for r in range(-R, R):
        for n in range(0, k):
            # theta = np.pi/2 - (np.pi*n)/(2*k)
            # fai = N*2*np.pi*n/k
            if n == 0:
                z_coordinate = int(0)
                x_coordinate = int(0)
                y_coordinate = int(r)
            else:
                theta = n * np.pi * np.pi / (4 * N * N)
                fai = n * (np.pi / N) / (np.sin(theta))
                x_coordinate = r * np.sin(theta) * np.cos(fai)
                y_coordinate = r * np.sin(theta) * np.sin(fai)
                z_coordinate = r * np.cos(theta)
            # print(x_coordinate, y_coordinate, z_coordinate)
            if x_coordinate + mid_x >= nrrd_data.shape[0] - 1 or y_coordinate + mid_y >= nrrd_data.shape[1] - 1 or \
                    z_coordinate + mid_z >= nrrd_data.shape[2] - 1:
                continue
            else:
                # 插值求灰度
                xmim, ymin, zmin = math.floor(x_coordinate + mid_x), math.floor(y_coordinate + mid_y), \
                                   math.floor(z_coordinate + mid_z)
                xd = x_coordinate + mid_x - xmim
                yd = y_coordinate + mid_y - ymin
                zd = z_coordinate + mid_z - zmin
                c000, c100, c010, c001, c101, c011, c110, c111 = nrrd_data[xmim, ymin, zmin], \
                                                                 nrrd_data[xmim + 1, ymin, zmin], \
                                                                 nrrd_data[xmim, ymin + 1, zmin], \
                                                                 nrrd_data[xmim, ymin, zmin + 1], \
                                                                 nrrd_data[xmim + 1, ymin, zmin + 1], \
                                                                 nrrd_data[xmim, ymin + 1, zmin + 1], \
                                                                 nrrd_data[xmim + 1, ymin + 1, zmin], \
                                                                 nrrd_data[xmim + 1, ymin + 1, zmin + 1]
                rotate_slice[r + R, n] = c000 * (1 - xd) * (1 - yd) * (1 - zd) + \
                                         c100 * xd * (1 - yd) * (1 - zd) + c010 * (1 - xd) * yd * (1 - zd) + \
                                         c001 * (1 - xd) * (1 - yd) * zd + c101 * xd * (1 - yd) * zd + \
                                         c011 * (1 - xd) * yd * zd + c110 * xd * yd * (1 - zd) + c111 * xd * yd * zd

    rotate_slice_result = (rotate_slice-rotate_slice.min())/(rotate_slice.max()-rotate_slice.min())

    cv2.imwrite(DATA_DIR_NEW + nrrd_filename[-15:-5] + '_' + trans + '_' + '.png', rotate_slice_result * 255)
    return rotate_slice


image_file = './data/npy/image/'
mask_file = './data/npy/mask/'
DATA_DIR_NEW = './data/data_rotate_{}/'.format(modal)
# DATA_DIR_NEW = 'F:/ruijin_Lymph_node/data_rotate_{}/'.format(modal)
if not os.path.exists(DATA_DIR_NEW):
    os.makedirs(DATA_DIR_NEW)

# DATA_IMAGE_LIST = './label/original/TP53_3D_{}.txt'.format(modal)
DATA_IMAGE_LIST = 'label_v2_class2_167_196.txt'
# DATA_IMAGE_LIST = './version9/label/buchong.txt'
dataset = DataSet(image_file=image_file, mask_file=mask_file, image_list_file=DATA_IMAGE_LIST, data_dir_new=DATA_DIR_NEW)
# print(dataset.data_length())
for idx in range(0, dataset.data_length()):
    print(idx)
    im1 = dataset.data_new(idx)
