import torch.nn as nn
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from read_data import DataSet_two_stream
from k_fold import k_fold_pre
from model.resnet import two_stream_resnet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
# used for logging to TensorBoard
from utils import accuracy, AverageMeter, evaluation_metrics
from lifelines.utils import concordance_index

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--model', default='resnet18', type=str, help='baseline of the model')
parser.add_argument('--pretrained', default=True, help='load pretrained model')
parser.add_argument('--fold', default=0, type=int, help='index of k-fold')  # 5-fold
parser.add_argument('--times', default=1, type=int, help='numbers of augmentation times')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--num_classes', default=2, type=int, help='numbers of classes (default: 1)') # num_classes 自己加的
parser.add_argument('--seed', default=1111, type=int, help='random seed(default: 1)')  # 随机数种子
parser.add_argument('--modarity', default='a', type=str, help='name of modarity(a,v)')
parser.add_argument('--resume', default='/home/data/chenxiahan/result/lymph-node-metastasis-classification/version4/try/checkpoint/',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='try',
                    type=str, help='name of experiment')
parser.add_argument('--model-dir', default='/home/data/chenxiahan/result/lymph-node-metastasis-classification/version4/',
                    type=str, help='name of model_dir')
parser.add_argument('--data-dir-rotate-a', default='/home/data/ruijin_Lymph_node/data_rotate_a_40/',
                    type=str,  help='name of data_dir_a')
parser.add_argument('--data-dir-rotate-v', default='/home/data/ruijin_Lymph_node/data_rotate_v_40/',
                    type=str, help='name of data_dir_v')
parser.add_argument('--data-dir-projection-a', default='/home/data/ruijin_Lymph_node/data_rotate_a_40/',
                    type=str,  help='name of data_dir_a')
parser.add_argument('--data-dir-projection-v', default='/home/data/ruijin_Lymph_node/data_rotate_v_40/',
                    type=str, help='name of data_dir_v')
parser.add_argument('--data-image-list', default='./label/label_v2_class2_143_157.txt',
                    type=str, help='name of data_dir')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')

args = parser.parse_args()  # #####很重要
DATA_IMAGE_LIST = args.data_image_list
MODEL_DIR = args.model_dir
# DATA_IMAGE_LIST = './label/data_2Dslice.txt'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global use_cuda, writer

    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.seed > 0:
        seed_torch(args.seed)  # 固定随机数种子
    # create model

    if args.model == 'resnet18' or args.model == 'resnet34' or args.model == 'resnet50' or args.model == 'resnet101':
        model = two_stream_resnet(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    else:
        print('Please choose right model.')
        return 0
    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(filename=MODEL_DIR + "%s/data_fold.txt" % args.name,
                                        image_list_file=DATA_IMAGE_LIST,
                                        fold=5, num_classes=args.num_classes)
    # 读取第k个fold的数据
    kwargs = {'num_workers': 0, 'pin_memory': True}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 和ImageNet数据集的保持一致

    train_datasets = DataSet_two_stream(option=args, image_list_file=DATA_IMAGE_LIST,
                                         fold=train_names[args.fold], times=args.times, shuffle=True,
                                         transform=transforms.Compose([# transforms.Resize(args.size),
                                                                       transforms.ToTensor(),
                                                                       normalize]))  # transforms.Resize(224),)
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=False,
                                               **kwargs)

    val_datasets = DataSet_two_stream(option=args, image_list_file=DATA_IMAGE_LIST,
                                       fold=val_names[args.fold], times=args.times, shuffle=True,
                                       transform=transforms.Compose([# transforms.Resize(args.size),
                                                                     transforms.ToTensor(),
                                                                     normalize]))  # transforms.Resize(224),)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size, shuffle=False, **kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar')
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 0
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return 0
    # define loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')  # 多分类用CrossEntropyLoss 原始网络中的sigmoid层改为softmax

    epoch = args.start_epoch - 1
    # evaluate on validation set
    for v in range(2):

        if v == 0:
            filename = MODEL_DIR + "{}/result/fold_{}_result_train_voting.txt".format(args.name, str(args.fold))
            image_names, output_roc, target_roc, auc, acc, precision, recall, f1score, sensitivity, specificity = \
                validate(train_loader, model, criterion, epoch, args.fold)
        else:
            filename = MODEL_DIR + "{}/result/fold_{}_result_test_voting.txt".format(args.name, str(args.fold))

            image_names, output_roc, target_roc, auc, acc, precision, recall, f1score, sensitivity, specificity = \
                validate(val_loader, model, criterion, epoch, args.fold)

            # acc, id_total, id_out, id_target = acc_voting(output_roc, target_roc, image_names)
            # ci = c_index(output_roc, target_roc)
            # resultname = MODEL_DIR + "{}/result/result_test_voting.txt".format(args.name)
            # resultfile = open(resultname, 'a')
            # resultfile.write(str(args.fold) + ' ' + str(val_acc.avg) + ' ' + str(AUROC[0]) + ' ' + str(ci) + ' ' + str(acc) + '\n')

            resultname = MODEL_DIR + "{}/result/result_test_evaluate.txt".format(args.name)
            resultfile = open(resultname, 'a')
            if args.fold == 0:
                resultfile.write("evaluate\tacc\tauc\tprecision\trecall\tf1score\tsensitivity\tspecificity\n")
            resultfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(args.fold, acc, auc, precision, recall, f1score, sensitivity, specificity))
            resultfile.close()

            # resultidname = MODEL_DIR + "{}/result/id_result_test.txt".format(args.name)
            # resultidfile = open(resultidname, 'a')
            # for id_index in range(len(id_out)):
            #     resultidfile.write(id_total[id_index] + ' ' + str(id_out[id_index]) + ' ' + str(id_target[id_index]) + '\n')
            # resultidfile.close()

        file = open(filename, 'w')  # 'a'  'w+' 追加
        label_val = torch.argmax(target_roc, -1)
        # print('image_names_len:', len(image_names))
        # print('output_roc.shape:', output_roc.shape[0])
        for i in range(output_roc.shape[0]):
            file.write(image_names[i] + ' ')
            for j in range(args.num_classes):
                file.write(str(output_roc[i][j].item()))
                file.write(' ')
            file.write(str(np.argmax(output_roc[i], -1).item()))
            file.write(' ')
            file.write(str(int(label_val[i].item())))
            file.write('\n')
        file.close()


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""

    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要
    target_roc = torch.zeros((0, args.num_classes))
    target_roc = target_roc.type(torch.LongTensor)
    image_names = []
    output_roc = torch.zeros((0, args.num_classes))
    with torch.no_grad():
        with tqdm(val_loader, ncols=130) as t:
            for i, (input1_a, input1_v, input2_a, input2_v, target, names) in enumerate(t):
                t.set_description("valid epoch %s" % epoch)
                if use_cuda:
                    target = target.type(torch.LongTensor).cuda()
                    input1_a = input1_a.type(torch.FloatTensor).cuda()
                    input1_v = input1_v.type(torch.FloatTensor).cuda()
                    input2_a = input2_a.type(torch.FloatTensor).cuda()
                    input2_v = input2_v.type(torch.FloatTensor).cuda()

                output, _, _ = model(input1_a, input1_v, input2_a, input2_v)
                image_names.extend(names)
                # measure accuracy and record loss
                target_class = torch.argmax(target, -1)
                val_loss = criterion(output, target_class)

                val_losses.update(val_loss.item(), target.size(0))

                output_softmax = F.softmax(output, dim=1)
                target_roc = torch.cat((target_roc, target.detach().cpu()), dim=0)
                output_roc = torch.cat((output_roc, output_softmax.detach().cpu()), dim=0)

                # -------------------------------------Accuracy--------------------------------- #
                acc = accuracy(output_softmax.detach(), target)  # 一个batchsize中n类的平均准确率  输出为numpy类型

                val_acc.update(acc, target.size(0))

                # measure elapsed time
                t.set_postfix({
                    'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                    'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
                )

    # -------------------------------------AUROC------------------------------------ #
    AUROC = aucrocs(output_roc, target_roc, args.num_classes)
    print('The AUROC is ', AUROC)
    # -------------------------------------AUROC------------------------------------ #
    auc, acc, precision, recall, f1score, sensitivity, specificity = evaluation_metrics(target_roc, output_roc)
    return image_names, output_roc, target_roc, auc, acc, precision, recall, f1score, sensitivity, specificity


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
    return AUROCs


def acc_voting(output, target, names):
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    output_arg = np.argmax(output_np, axis=1)
    target_arg = np.argmax(target_np, axis=1)
    # 把病人的输出统计为一个

    total_slice_num = len(output_arg)  # 总的测试案例数量 multi-slice
    i_names = []
    for a in names:
        a = a[0:10]
        i_names.append(a)
    # 统计每个id有多少层 （按顺序输出
    ID_number = len(set(i_names))  # 不重复的ID数量
    id_total = []
    number_total = []
    id_i = i_names[0]
    id_total.append(id_i)
    flag = 0
    for num in range(total_slice_num):
        if i_names[num] == id_i:
            flag = flag + 1
        else:
            number_total.append(flag)
            id_i = i_names[num]  # 统计下一个id
            id_total.append(id_i)
            flag = 1
        if num == total_slice_num - 1:  # 最后一个slice
            number_total.append(flag)

    id_target = np.zeros([ID_number])
    id_out = np.zeros([ID_number])
    for i in range(ID_number):  # 截取相同ID的片段
        if i == 0:
            out_i = output_arg[0:number_total[0]]
            target_i = target_arg[0:number_total[0]]
        else:
            sum_slice = sum(number_total[:i])
            out_i = output_arg[sum_slice:number_total[i]+sum_slice]
            target_i = target_arg[sum_slice:number_total[i]+sum_slice]

        id_target[i] = target_i[0]

        count_out_i = np.zeros([args.num_classes])  # 统计每个元素出现的次数
        for c in range(args.num_classes):
            count_out_i[c] = sum(out_i == c)

        if args.num_classes == 2:
            if count_out_i[0] == count_out_i[1]:  # 两类数量相等
                id_out[i] = 1  # 判断为N2
            else:
                id_out[i] = np.argmax(np.bincount(out_i))  # 出现次数最多的元素
        if args.num_classes == 3:
            if count_out_i[0] == count_out_i[1] and count_out_i[0] >= count_out_i[2]:
                id_out[i] = 0  # 判断为N0
            elif count_out_i[0] == count_out_i[2] and count_out_i[0] >= count_out_i[1]:
                id_out[i] = 2  # 判断为N0
            elif count_out_i[1] == count_out_i[2] and count_out_i[1] >= count_out_i[0]:
                id_out[i] = 2  # 判断为N2
            else:
                id_out[i] = np.argmax(np.bincount(out_i))

    print(id_out)
    print(id_target)
    acc = sum(id_out == id_target)/ID_number
    print(acc)
    return acc, id_total, id_out, id_target


def c_index(output, target):

    """
    Returns:
    c_index of all classes.
    """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    output_arg = output_np[:, 1]
    target_arg = target_np[:, 1]

    ci = concordance_index(target_arg.reshape(-1), output_arg.reshape(-1))
    return ci


if __name__ == '__main__':

    main()

