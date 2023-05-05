"""
 为了保证映射图每一行都可以填满 y固定 x等距离变化，求出对应的fai和lamda
 R动态变化
 共转了Q圈
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import cv2
import copy
import os
import time


def mercator_projection(image, mid, num_lines, N, Rmax, Q, save_name):
    # 为了保证映射图每一行都可以填满 y固定 x等距离变化，求出对应的fai和lamda
    out = np.zeros([2*num_lines, 2*N*Q])
    mid_x, mid_y, mid_z = mid[0], mid[1], mid[2]
    fai_max = np.pi / 2. * 8/9

    x_fixed = np.linspace(0, 2*np.pi*Rmax*Q, 2*N*Q)

    for k in range(2*N*Q):

        # calculate fai and lamda
        lamda = x_fixed[k] / Rmax
        # R = -Rmax / (2 * np.pi * Q) * lamda + Rmax
        R = (3-Rmax) * lamda * lamda / ((2 * np.pi * Q) ** 2) + Rmax
        if R == 0:
            x_coordinate = 0
            y_coordinate = 0
            z_coordinate = 0
        else:
            ymax_R = (np.log((1 + np.sin(fai_max)) / (1 - np.sin(fai_max))) / 2) * R
            y_fixed = np.linspace(-ymax_R, ymax_R, 2 * num_lines)

            fai = 2 * np.arctan(np.exp(y_fixed / R)) - np.pi / 2

            x_coordinate = R * np.cos(fai) * np.cos(lamda)
            y_coordinate = R * np.cos(fai) * np.sin(lamda)
            z_coordinate = R * np.sin(fai)

        # 插值求灰度
        xmim, ymin, zmin = np.floor(x_coordinate + mid_x).astype(int), np.floor(y_coordinate + mid_y).astype(int), \
                           np.floor(z_coordinate + mid_z).astype(int)
        xd = x_coordinate + mid_x - xmim
        yd = y_coordinate + mid_y - ymin
        zd = z_coordinate + mid_z - zmin
        c000, c100, c010, c001, c101, c011, c110, c111 = image[zmin, xmim, ymin], image[zmin, xmim + 1, ymin], \
                                                         image[zmin, xmim, ymin + 1], image[zmin + 1, xmim, ymin], \
                                                         image[zmin + 1, xmim + 1, ymin], image[zmin + 1, xmim, ymin + 1], \
                                                         image[zmin, xmim + 1, ymin + 1], image[zmin + 1, xmim + 1, ymin + 1]
        value = c000 * (1 - xd) * (1 - yd) * (1 - zd) + \
                c100 * xd * (1 - yd) * (1 - zd) + \
                c010 * (1 - xd) * yd * (1 - zd) + \
                c001 * (1 - xd) * (1 - yd) * zd + \
                c101 * xd * (1 - yd) * zd + \
                c011 * (1 - xd) * yd * zd + \
                c110 * xd * yd * (1 - zd) + \
                c111 * xd * yd * zd
        # x_matrix, y_matrix = np.floor(x).astype(int), np.floor(y).astype(int)
        out[:, k] = value
    cv2.imwrite(save_name, out)
    # plt.imshow(out,'gray')

    # ell1 = Ellipse(xy=(80-1, 40), width=80*2, height=40*2, angle=0, alpha=0.3)
    # ax.add_artist(ell1)
    # plt.show()
    return out


def image_transformation(im1, mask, angle):
    image1 = copy.deepcopy(im1)

    zz, xx, yy = np.nonzero(mask!=0)
    mid_x1 = int(round((max(xx) + min(xx)) / 2))
    mid_y1 = int(round((max(yy) + min(yy)) / 2))

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
    elif angle == 'fh' or angle == 'flip_horizontally':  # 水平翻转
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.flip(image1[i1, :, :], 1)
    elif angle == 'fv' or angle == 'flip_vertically':  # 垂直翻转
        for i1 in range(image1.shape[0]):
            image1[i1, :, :] = cv2.flip(image1[i1, :, :], 0)
    else:
        print('please choose right angle')

    return image1


if __name__ == '__main__':
    modarity = 'a'
    image_path = 'F:/ruijin_Lymph_node/npy/image/'
    mask_path = 'F:/ruijin_Lymph_node/npy/mask/'
    save_path = 'F:/ruijin_Lymph_node/MP/Mercator_Projection2{}_try/'.format(modarity)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    names = os.listdir(image_path)
    num_lines = 120
    N = 36
    Q = 5
    # imagesize = [2 * num_lines, 2 * N * Q]
    # augmentation = ['00', 'r10', 'r20', 'r30', 'r40', 'r50', 'r60', 'r70', 'r80', 'r90', 'r100',
    #                 'r110', 'r120', 'r130', 'r140', 'r150', 'r160', 'r170', 'r180', 'r190', 'r200',
    #                 'r210', 'r220', 'r230', 'r240', 'r250', 'r260', 'r270', 'r280', 'r290', 'r300',
    #                 'r310', 'r320', 'r330', 'r340', 'r350', 'fh00', 'fhr10', 'fhr20', 'fhr30', 'fhr40',
    #                 'fhr50', 'fhr60', 'fhr70', 'fhr80', 'fhr90', 'fhr100','fhr110', 'fhr120', 'fhr130',
    #                 'fhr140', 'fhr150', 'fhr160', 'fhr170', 'fhr180']
    # augmentation = ['fh00', 'fhr10', 'fhr20', 'fhr30', 'fhr40', 'fhr50', 'fhr60', 'fhr70', 'fhr80', 'fhr90', 'fhr100',
    #                 'fhr110', 'fhr120', 'fhr130', 'fhr140', 'fhr150', 'fhr160', 'fhr170', 'fhr180', 'fhr190', 'fhr200',
    #                 'fhr210', 'fhr220', 'fhr230', 'fhr240', 'fhr250', 'fhr260', 'fhr270', 'fhr280', 'fhr290', 'fhr300',
    #                 'fhr310', 'fhr320', 'fhr330', 'fhr340', 'fhr350']
    augmentation = ['fv00', 'fvr10', 'fvr20', 'fvr30', 'fvr40', 'fvr50', 'fvr60', 'fvr70', 'fvr80', 'fvr90', 'fvr100',
                    'fvr110', 'fvr120', 'fvr130', 'fvr140', 'fvr150', 'fvr160', 'fvr170', 'fvr180', 'fvr190', 'fvr200',
                    'fvr210', 'fvr220', 'fvr230', 'fvr240', 'fvr250', 'fvr260', 'fvr270', 'fvr280', 'fvr290', 'fvr300',
                    'fvr310', 'fvr320', 'fvr330', 'fvr340', 'fvr350', 'fhr190', 'fhr200', 'fhr210', 'fhr220', 'fhr230',
                    'fhr240', 'fhr250', 'fhr260', 'fhr270', 'fhr280', 'fhr290', 'fhr300',
                    'fhr310', 'fhr320', 'fhr330', 'fhr340', 'fhr350']
    print('augmentation times:', len(augmentation))
    path = 'Mercator_Projection_labelv3{}.txt'.format(modarity)

    # file = open(path, 'w')
    for index in range(680, len(names)):  # len(names)
        print(index)
        name = names[index]
        if name.endswith('{}.npy'.format(modarity)):
            image = np.load(image_path+name)
            mask = np.load(mask_path+name)

            image_trans_mid_fv = image_transformation(image, mask, angle='fv')
            mask_trans_mid_fv = image_transformation(mask, mask, angle='fv')

            image_trans_mid_fh = image_transformation(image, mask, angle='fh')
            mask_trans_mid_fh = image_transformation(mask, mask, angle='fh')

            # zz, xx, yy = mask.nonzero()
            # mid_x = int(round((max(xx) + min(xx)) / 2))
            # mid_y = int(round((max(yy) + min(yy)) / 2))
            # mid_z = int(round((max(zz) + min(zz)) / 2))
            # dx = xx.max()-xx.min()
            # dy = yy.max()-yy.min()
            # dz = zz.max()-zz.min()
            # mid = [mid_x, mid_y, mid_z]
            # Rmax = min([dx, dy, dz]) / 2
            # print(name, Rmax)
            # file.write(name[:-5])
            for angle in augmentation:
                # start = time.time()
                if angle == '00':
                    image_trans = copy.deepcopy(image)
                    mask_trans = copy.deepcopy(mask)
                # elif angle[0:2] == 'fh' or angle[0:2] == 'fv':
                #     image_trans_mid = image_transformation(image, mask, angle=angle[0:2])
                #     mask_trans_mid = image_transformation(mask, mask, angle=angle[0:2])
                #     if angle[2:] == '00':
                #         image_trans = copy.deepcopy(image_trans_mid)
                #         mask_trans = copy.deepcopy(mask_trans_mid)
                #     else:
                #         image_trans = image_transformation(image_trans_mid, mask_trans_mid, angle=angle[2:])
                #         mask_trans = image_transformation(mask_trans_mid, mask_trans_mid, angle=angle[2:])
                elif angle[0:2] == 'fv':
                    if angle[2:] == '00':
                        image_trans = copy.deepcopy(image_trans_mid_fv)
                        mask_trans = copy.deepcopy(mask_trans_mid_fv)
                    else:
                        image_trans = image_transformation(image_trans_mid_fv, mask_trans_mid_fv, angle=angle[2:])
                        mask_trans = image_transformation(mask_trans_mid_fv, mask_trans_mid_fv, angle=angle[2:])
                elif angle[0:2] == 'fh':
                    if angle[2:] == '00':
                        image_trans = copy.deepcopy(image_trans_mid_fh)
                        mask_trans = copy.deepcopy(mask_trans_mid_fh)
                    else:
                        image_trans = image_transformation(image_trans_mid_fh, mask_trans_mid_fh, angle=angle[2:])
                        mask_trans = image_transformation(mask_trans_mid_fh, mask_trans_mid_fh, angle=angle[2:])
                else:
                    image_trans = image_transformation(image, mask, angle=angle)
                    mask_trans = image_transformation(mask, mask, angle=angle)

                zz, xx, yy = np.nonzero(mask_trans)
                mid_x = int(round((max(xx) + min(xx)) / 2))
                mid_y = int(round((max(yy) + min(yy)) / 2))
                mid_z = int(round((max(zz) + min(zz)) / 2))
                dx = xx.max() - xx.min()
                dy = yy.max() - yy.min()
                dz = zz.max() - zz.min()
                mid = [mid_x, mid_y, mid_z]
                Rmax = min([dx, dy]) / 2
                print(name, Rmax)
                if Rmax < 3:
                    continue
                # file.write(name)

                save_name = save_path + name[:-4] + '-' + angle + '.png'
                mercator_projection(image_trans, mid, num_lines, N, Rmax, Q, save_name)

                # save_name = save_path + name[:-4] + '-' + angle + '-mask.png'
                # mercator_projection(mask_trans * 255, mid, num_lines, N, Rmax, Q, save_name)
                # print('time:', time.time()-start)
                # file.write(' ' + str(int(Rmax)) + '-' + angle)
            # file.write('\n')
    # file.close()
