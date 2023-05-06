import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn as nn
import torch
import random
import os
import torch.nn.functional as F
from model.utils import Normalize


def evaluation_metrics(target, output):
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    auc = roc_auc_score(target_np[:, 0], output_np[:, 0])

    pred = np.argmax(output_np, axis=1)
    label = np.argmax(target_np, axis=1)
    right = sum(pred == label)
    acc = right / output.shape[0]

    #  mean class
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1score = f1_score(label, pred, average='macro')

    # -----TN----- #
    TN = np.logical_and(pred == 0, label == 0).sum()
    FN = np.logical_and(pred == 0, label == 1).sum()
    FP = np.logical_and(pred == 1, label == 0).sum()
    TP = np.logical_and(pred == 1, label == 1).sum()
    print('TN:', TN, 'FN:', FN, 'FP:', FP, 'TP:', TP)
    # -----sensitivity=recall----- #
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return auc, acc, precision, recall, f1score, sensitivity, specificity



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output, target):
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    output_arg = np.argmax(output_np, axis=1)
    target_arg = np.argmax(target_np, axis=1)
    right = sum(output_arg == target_arg)
    acc = right / output.shape[0]

    return acc


def aucrocs(output, target, num_classes):

    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    AUROCs=[]
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(target_np[:, i], output_np[:, i]))
    return np.mean(AUROCs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class loss_function:
    def __init__(self, weight_decay_fc=0, p=1, weight_decay_ssl=0):
        self.weight_decay = weight_decay_fc
        self.p = p
        self.augmentselfsupervised = Augment_BatchCriterion()
        self.weight_decay_ssl = weight_decay_ssl
        # weight = torch.FloatTensor([1.5,1,1.5]).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.l2norm = Normalize(2)

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''

        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, self.p)
            reg_loss = reg_loss + l2_reg

        reg_loss = self.weight_decay * reg_loss

        return reg_loss

    def calculate_loss(self, model, output, target):
        loss_cro = self.criterion(output, target)
        weight_list = self.get_weight(model.classifier)
        loss = loss_cro + self.regularization_loss(weight_list)

        return loss

    def calculate_augment_ssl_loss(self, model, output, out_feature, trans_feature, target):
        """
        计算扩增样本间自监督约束  len(out_a)>=2
        :param model:
        :param output: [bs, num_class]
        :param out_feature:  [bs, feature_num]
        :param aug_feature:  [bs, aug_times, feature_num]
        :param target: [bs]
        :return:
        """
        loss_cro = self.criterion(output, target)
        weight_list = self.get_weight(model.classifier)

        selfsupervised_loss = self.augmentselfsupervised(None, None, out_feature, target, mode='neg')

        # 两两之间求相似
        tmp = 0
        for i in range(0, trans_feature.shape[1]):
            for j in range(i+1, trans_feature.shape[1]):
                timei = trans_feature[:, i, :]
                timej = trans_feature[:, j, :]
                selfsupervised_loss = selfsupervised_loss + self.augmentselfsupervised(timei, timej, None, target, mode='pos')
                tmp = tmp + 1

        loss = loss_cro + self.regularization_loss(weight_list) + self.weight_decay_ssl * selfsupervised_loss/ tmp
        return loss


class Augment_BatchCriterion(nn.Module):
    def __init__(self, batchSize=16, T=0.1):
        super(Augment_BatchCriterion, self).__init__()
        self.T = T  # temperature parameter for softmax

        # self.batchSize = batchSize

    def forward(self, x_a, x_v, out_feature, target, mode='all'):

        # diag_mat = 1 - torch.eye(batchSize).cuda()
        # # get positive innerproduct
        # # x_a = x.narrow(0, 0, batchSize)  # [16, 128]
        # # x_v = x.narrow(0, batchSize, batchSize)  # 作为基准fv  # [16, 128]
        # # 后mini-batch个 和 前mini-batch个 分别提取出来 x_a和对应的x_v是positive pair
        # # reordered_x = reordered_x.data   reordered_x和对应的x是positive pair
        if mode is not 'neg':
            batchSize = x_a.shape[0]
            pos = (x_a * x_v.data).sum(1).div_(self.T).exp_()  # exp(sum/T)  # torch.Size([16])
            pos_all_prob = torch.mm(x_a, x_v.t().data).div_(self.T).exp_()  # ai对每一个vk求和（余弦距离）
            pos_all_div = pos_all_prob.sum(1)
            pos_prob = torch.div(pos, pos_all_div)
            loss_pos = pos_prob.log_()
            lnpossum = loss_pos.sum(0)
            loss_pos = - lnpossum / batchSize

        if mode is not 'pos':
            batchSize = out_feature.shape[0]
            # negative probability, remove diag  去掉自己*自己  去掉相同类别
            target = target.unsqueeze(1).float()
            same_class = torch.mm(target+1, (target+1).t().data) == 2
            # neg = torch.mm(x_v, x_v.t().data).div_(self.T).exp_() * diag_mat
            neg = torch.mm(out_feature, out_feature.t().data).div_(self.T).exp_() * same_class.float()
            neg_all_prob = torch.mm(out_feature, out_feature.t().data).div_(self.T).exp_()
            neg_div = neg_all_prob.sum(1)
            neg_div = neg_div.repeat(batchSize, 1)
            neg_prob = torch.div(neg, neg_div)

            neg_prob = -neg_prob.add(-1)
            loss_neg = neg_prob.log_().sum(1)

            lnnegsum = loss_neg.sum(0)
            loss_neg = - lnnegsum / batchSize

        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        else:
            return loss_pos + loss_neg
