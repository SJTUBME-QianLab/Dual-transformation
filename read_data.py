import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import random


class DataSet_two_stream(Dataset):
    def __init__(self, option, image_list_file, fold, times=1, transform=None, shuffle=False):
        """
        Args:
            image_list_file: path to the file containing images
                with corresponding labels.
            fold=train_names[args.fold]
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        # self.angle_dict_rotate = {1: '-00', 2: '-fv00', 3: '-fh00'}
        self.angle_dict_rotate = {1: '_000000_',  2: '_000hor_', 3: '_000ver_', 4: '_l40000_', 5: '_l40hor_', 6: '_l40ver_',
                                  7: '_r40_', 8: '_r40hor_', 9: '_r40ver_', 10: '_l20000_', 11: '_l20hor_', 12: '_l20ver_',
                                  13: '_r20_', 14: '_r20hor_', 15: '_r20ver_'}

        self.angle_dict_projection = {1: '-00', 2: '-fv00', 3: '-fh00', 4: '-r90', 5: '-fvr90', 6: '-fhr90', 7: '-r180',
                                      8: '-fvr180', 9: '-fhr180', 10: '-r270', 11: '-fvr270', 12: '-fhr270'}
        # self.angle_dict_projection = {1: '_0_0', 2: '_90_0', 3: '_90_270',  4: '_45_0', 5: '_90_45', 6: '_45_90',
        #                               7: '_90_135', 8: '_315_0', 9: '_315_90', 10: '_15_15', 11: '105_15', 12: '_105_285'}  # slice

        fileline = open(image_list_file, "r").readlines()
        aug_rotate, aug_projection = [], []
        for line_index in fold:
            line = fileline[line_index]
            items = line.split()
            label = items[1:3]
            if label[0] == '0':
                label[1] = '1'
            label = [int(i) for i in label]

            for j in range(times):
                image_names.append(items[0])
                aug_rotate.append(self.angle_dict_rotate[j+1])
                aug_projection.append(self.angle_dict_projection[j+1])
                labels.append(label)

        # 手动打乱扩增的顺序
        # if shuffle is True:
        #     image_num = len(image_names)
        #     self.aug_rotate = random.sample(aug_rotate, image_num)
        #     self.aug_projection = random.sample(aug_projection, image_num)
        # else:
        self.aug_rotate = aug_rotate
        self.aug_projection = aug_projection

        self.image_names = image_names
        self.labels = labels
        self.option = option
        self.transform = transform
        self.times = times

    def __getitem__(self, index):    # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        # print(image_name)
        random_trans1 = random.randint(1, self.times)
        image_rotate_a = Image.open(self.option.data_dir_rotate_a + image_name + self.angle_dict_rotate[random_trans1] + '.png').convert('RGB')
        image_rotate_v = Image.open(self.option.data_dir_rotate_v + image_name + self.angle_dict_rotate[random_trans1] + '.png').convert('RGB')

        random_trans2 = random.randint(1, self.times)
        image_projection_a = Image.open(
            self.option.data_dir_projection_a + image_name + 'a' + self.angle_dict_projection[random_trans2] + '.png').convert('RGB')
        image_projection_v = Image.open(
            self.option.data_dir_projection_v + image_name + 'v' + self.angle_dict_projection[random_trans2] + '.png').convert('RGB')

        label = torch.Tensor(self.labels[index])  # 存储标签

        if self.transform is not None:
            image_rotate_a = self.transform(image_rotate_a)
            image_rotate_v = self.transform(image_rotate_v)
            image_projection_a = self.transform(image_projection_a)
            image_projection_v = self.transform(image_projection_v)

        return image_rotate_a, image_rotate_v, image_projection_a, image_projection_v, label, image_name

    def __len__(self):
        return len(self.image_names)