import argparse
import os
import random
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
# from pandas import np
import numpy as np
from read_data import DataSet_two_stream
from k_fold import k_fold_pre
from model.resnet import two_stream_resnet
import torch.nn.functional as F
import math
from tqdm import tqdm
# used for logging to TensorBoard
from tensorboardX import SummaryWriter
from utils import *
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--model', default='resnet18', type=str, help='baseline of the model')
parser.add_argument('--pretrained', default=True, type=bool, help='load pretrained model')
parser.add_argument('--fold', default=0, type=int, help='index of k-fold')
parser.add_argument('--times', default=3, type=int, help='numbers of augmentation times')
parser.add_argument('--n_epoch', default=20, type=int, help='number of epoch to change')
# parser.add_argument('--size', default=64, type=int, help='resize input image')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--num_classes', default=2, type=int, help='numbers of classes (default: 1)')  # num_classes 自己加的
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer (SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  # 正则化参数
                    help='weight decay (default: 1e-4)')
parser.add_argument('--weight-decay-fc', '--wdfc', default=0.01, type=float,  # 全连接层正则化参数
                    help='weight decay of classifier (default: 1e-4)')
parser.add_argument('--weight-decay-ssl', default=0.01, type=float, help='自监督约束项的系数 (default: 0.01)')
parser.add_argument('--growth', default=32, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--seed', default=1111, type=int, help='random seed(default: 1)')
parser.add_argument('--resume', default='./data/result/lymph-node-metastasis-classification/version6/'
                                        'try/checkpoint/',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='try',
                    type=str, help='name of experiment')
parser.add_argument('--model-dir', default='./data/result/lymph-node-metastasis-classification/version6/',
                    type=str, help='name of model_dir')
parser.add_argument('--data-dir-rotate-a', default='./data/ruijin_Lymph_node/data_rotate_a_40/',
                    type=str,  help='name of data_dir_a')
parser.add_argument('--data-dir-rotate-v', default='./data/ruijin_Lymph_node/data_rotate_v_40/',
                    type=str, help='name of data_dir_v')
parser.add_argument('--data-dir-projection-a', default='./data/ruijin_Lymph_node/Mercator_Projection/Mercator_Projection2a/',
                    type=str,  help='name of data_dir_a')
parser.add_argument('--data-dir-projection-v', default='./data/ruijin_Lymph_node/Mercator_Projection/Mercator_Projection2v/',
                    type=str, help='name of data_dir_v')
parser.add_argument('--data-image-list', default='./label/label.txt',type=str, help='name of data_dir')
parser.add_argument('--tensorboard', default=True,  help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True, help='whether to use_cuda(default: True)')
# DATA_DIR = 'F:/ruijin Lymph node/2D_slice128x128_max_layer/'
# DATA_IMAGE_LIST = './label/data_2Dslice.txt'

args = parser.parse_args()  # #####很重要

DATA_IMAGE_LIST = args.data_image_list
MODEL_DIR = args.model_dir
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.model_dir + args.name + "/result/"):
    os.makedirs(args.model_dir + args.name + "/result/")

io = IOStream(args.model_dir + args.name + '/result/' + '/run.log')
io.cprint(str(args))


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

    if args.tensorboard:
        writer = SummaryWriter(MODEL_DIR+args.name)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.seed > 0:
        seed_torch(args.seed)  # 固定随机数种子
    # create model

    if args.model == 'resnet18' or args.model == 'resnet34' or args.model == 'resnet50' or args.model == 'resnet101':
        model = two_stream_resnet(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    else:
        print('Please choose right model.')
        return 0

    # print(model)
    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()

    torch.save({'state_dict': model.state_dict()}, MODEL_DIR+"%s/checkpoint_init.pth.tar" % args.name)
    # print("=> not load pretrained checkpoint")

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(filename=MODEL_DIR+"%s/data_fold.txt" % args.name, image_list_file=DATA_IMAGE_LIST,
                                        fold=5, num_classes=args.num_classes)

    if not os.path.exists(MODEL_DIR+"%s/result/" % (args.name)):
        os.makedirs(MODEL_DIR+"%s/result/" % (args.name))
    filelossacc_name = MODEL_DIR+"{}/result/train_fold{}_loss_acc.txt".format(args.name, args.fold)
    filelossacc = open(filelossacc_name, 'a')
    best_prec = 0  # 第k个fold的准确率
    # 读取第k个fold的数据
    kwargs = {'num_workers': 0, 'pin_memory': True}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 和ImageNet数据集的保持一致
    # transforms_list = [transforms.RandomRotation([-10, 10]),
    #                    transforms.RandomHorizontalFlip(p=0.5),
    #                    transforms.RandomVerticalFlip(p=0.5)
    #                    ]

    train_datasets = DataSet_two_stream(option=args, image_list_file=DATA_IMAGE_LIST,
                                         fold=train_names[args.fold], times=args.times, shuffle=True,
                                         transform=transforms.Compose([# transforms.RandomChoice(transforms_list),
                                                                       # transforms.Resize(args.size),
                                                                       transforms.ToTensor(),
                                                                       normalize]))  # transforms.Resize(224),)
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True, **kwargs)

    val_datasets = DataSet_two_stream(option=args, image_list_file=DATA_IMAGE_LIST,
                                       fold=val_names[args.fold], times=args.times, shuffle=True,
                                       transform=transforms.Compose([# transforms.Resize(args.size),
                                                                     transforms.ToTensor(),
                                                                     normalize]))  # transforms.Resize(224),))
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.resume:
        if os.path.isfile(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar')
            pretrained_dict = checkpoint['state_dict']
            # pretrained_dict.pop("classifier.weight")
            # pretrained_dict.pop("classifier.bias")
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            print("=> use initial checkpoint")
            checkpoint = torch.load(MODEL_DIR+"%s/checkpoint_init.pth.tar" % args.name)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return 0
    # define loss function
    # criterion = nn.BCELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='mean')  # 多分类用CrossEntropyLoss 原始网络中的sigmoid层改为softmax
    criterion = loss_function(weight_decay_fc=args.weight_decay_fc, p=1, weight_decay_ssl=args.weight_decay_ssl)
    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_losses, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.fold)

        for name, layer in model.named_parameters():
            # print('--name:', name, '--requires_grad:', layer.requires_grad)
            writer.add_histogram('fold' + str(args.fold) + '/' + name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram('fold' + str(args.fold) + '/' + name + '_data', layer.cpu().data.numpy(), epoch)
        # evaluate on validation set
        val_losses, val_acc, prec1, output_val, label_val, AUROC = validate(val_loader, model, criterion, epoch, args.fold)

        if args.tensorboard:
            # x = model.conv1.weight.data
            # x = vutils.make_grid(x, normalize=True, scale_each=True)
            # writer.add_image('data' + str(k) + '/weight0', x, epoch)  # Tensor
            writer.add_scalars('data' + str(args.fold) + '/loss',
                               {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
            writer.add_scalars('data' + str(args.fold) + '/Accuracy', {'train_acc': train_acc.avg, 'val_acc': val_acc.avg},
                               epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec
        if is_best == 1:
            best_prec = max(prec1, best_prec)  # 这个fold的最高准确率
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, is_best, epoch, args.fold)

        out_write = str(train_losses.avg) + ' ' + str(train_acc.avg) + ' ' + str(val_acc.avg) + '\n'
        filelossacc.write(out_write)
    writer.close()
    filelossacc.write('\n')
    filelossacc.close()


def train(train_loader, model, criterion, optimizer, epoch, fold):
    """Train for one epoch on the training set"""
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    # switch to train mode
    model.train()

    with tqdm(train_loader, ncols=130) as t:
        for i, (input1_a, input1_v, input2_a, input2_v, target, _) in enumerate(t):
            t.set_description("train epoch %s" % epoch)
            if use_cuda:
                target = target.type(torch.LongTensor).cuda()
                input1_a = input1_a.type(torch.FloatTensor).cuda()
                input1_v = input1_v.type(torch.FloatTensor).cuda()
                input2_a = input2_a.type(torch.FloatTensor).cuda()
                input2_v = input2_v.type(torch.FloatTensor).cuda()
            # batch_x_a : [192, 3, 80, 254]

            output, features, transform_features = model(input1_a, input1_v, input2_a, input2_v)

            # measure accuracy and record loss
            # one-hot 形式转化为类别标签
            target_class = torch.argmax(target, -1)
            # train_loss = criterion.calculate_loss(model, output, target_class)
            train_loss = criterion.calculate_augment_ssl_loss(model, output, features, transform_features, target_class)
            train_losses.update(train_loss.item(), target.size(0))

            output_softmax = F.softmax(output, dim=1)
            acc = accuracy(output_softmax.detach(), target)
            train_acc.update(acc, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t.set_postfix({
                'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=train_losses),
                'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=train_acc)}
            )

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_acc', train_acc.avg, epoch)
    return train_losses, train_acc


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要
    target_roc = torch.zeros((0, args.num_classes))
    target_roc = target_roc.type(torch.LongTensor)
    output_roc = torch.zeros((0, args.num_classes))
    with torch.no_grad():
        with tqdm(val_loader, ncols=130) as t:
            for i, (input1_a, input1_v, input2_a, input2_v, target, _) in enumerate(t):
                t.set_description("valid epoch %s" % epoch)
                if use_cuda:
                    target = target.type(torch.LongTensor).cuda()
                    input1_a = input1_a.type(torch.FloatTensor).cuda()
                    input1_v = input1_v.type(torch.FloatTensor).cuda()
                    input2_a = input2_a.type(torch.FloatTensor).cuda()
                    input2_v = input2_v.type(torch.FloatTensor).cuda()

                output, features, transform_features = model(input1_a, input1_v, input2_a, input2_v)

                # measure accuracy and record loss
                # one-hot 形式转化为类别标签
                target_class = torch.argmax(target, -1)
                val_loss = criterion.calculate_augment_ssl_loss(model, output, features, transform_features, target_class)
                val_losses.update(val_loss.item(), target.size(0))

                output_softmax = F.softmax(output, dim=1)
                target_roc = torch.cat((target_roc, target.detach().cpu()), dim=0)
                output_roc = torch.cat((output_roc, output_softmax.detach().cpu()), dim=0)

                # -------------------------------------Accuracy--------------------------------- #
                acc = accuracy(output_softmax.detach(), target)  # 一个batchsize中n类的平均准确率  输出为numpy类型
                val_acc.update(acc, target.size(0))

                t.set_postfix({
                    'loss': '{loss.val:.4f}({loss.avg:.4f})'.format(loss=val_losses),
                    'Acc': '{acc.val:.4f}({acc.avg:.4f})'.format(acc=val_acc)}
                )

    # -------------------------------------AUROC------------------------------------ #
    AUROC = aucrocs(output_roc, target_roc, args.num_classes)
    print('The AUROC is %.4f' % AUROC)
    # -------------------------------------AUROC------------------------------------ #

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_acc.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_AUC', AUROC, epoch)

    return val_losses, val_acc, val_acc.avg, output_roc, target_roc, AUROC


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # epoch = epoch - 150
    if epoch < args.n_epoch:
        # lr = args.lr * epoch / args.n_epoch
        lr = args.lr
    else:
        # lr = args.lr
    #     # lr = args.lr * (0.1 ** (epoch // 50))
    #     # lr = args.lr * 0.1 * (0.1 ** (epoch // 50))
    #     # lr = args.lr * (math.e ** (-epoch / args.n_epoch))
        lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.epochs)) / 2
    #     # if lr <= 0.000001:
    #     #     lr = 0.000001
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(args.fold) + '/learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = MODEL_DIR + "%s/checkpoint/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, MODEL_DIR+'%s/checkpoint/' % (args.name) + 'model_best' + str(fold) + '.pth.tar')


if __name__ == '__main__':

    main()

