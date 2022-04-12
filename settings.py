import sys
import argparse

def get_args():

    # default parameters
    clothing1m_root = "./data/Clothing-1M"
    mnist_root = './data/mnist'
    cifar10_root = './data/cifar10'
    batch_size = 32
    device = 'cuda:0'
    data_device = 1
    noise_type = 'sn'
    stage1 = 70
    stage2 = 200

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # training parameters group 1
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-5, type=float,
                        metavar='H-P', help='learning rate of stage3')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--backbone', dest="backbone", default="cnn", type=str,
                        help="backbone for9 training")
    parser.add_argument('--optim', dest="optim", default="Adam", type=str,
                        choices=['SGD', 'Adam'], help="Optimizer for training")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # training parameters group 2
    parser.add_argument('--forget-rate', '--fr', '--forget_rate', default=0.2, type=float,
                        metavar='H-P', help='Forget rate. Suggest same with noisy density.')
    parser.add_argument('--num-gradual', '--ng', '--num_gradual', default=10, type=int,
                        metavar='H-P', help='how many epochs for linear drop rate, can be 5, 10, 15.')
    parser.add_argument('--exponent', default=1, type=float,
                        metavar='H-P', help='exponent of the forget rate, can be 0.5, 1, 2.')
    parser.add_argument('--loss-type', dest="loss_type", default="cocorrecting_plus", type=str)
    parser.add_argument('--warmup', '--wm', '--warm-up', default=0, type=float,
                        metavar='H-P', help='Warm up process eopch, default 0.')
    parser.add_argument('--linear-num', '--linear_num', default=256, type=int,
                        metavar='H-P', help='linear layer feature num of the CNN model. Default is 256')
    # training parameters group 3
    parser.add_argument('--alpha', default=0.4, type=float,
                        metavar='H-P', help='the coefficient of Compatibility Loss')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='the coefficient of Entropy Loss')
    parser.add_argument('--lambda1', default=200, type=int,
                        metavar='H-P', help='the value of lambda, ')
    parser.add_argument('--K', default=10.0, type=float, )
    # training parameters group 4
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--stage1', default=stage1, type=int,
                        metavar='H-P', help='number of epochs utill stage1')
    parser.add_argument('--stage2', default=stage2, type=int,
                        metavar='H-P', help='number of epochs utill stage2')
    parser.add_argument('--save-all-y', default=False, action='store_true', help="Save y in all epoch, default false.")
    # Nosie settings
    parser.add_argument('--noise', default=0.20, type=float,
                        help='noise density of data label')
    parser.add_argument('--noise_type', default=noise_type,  choices=['clean', 'sn', 'pairflip'],type=str,
                        help='noise tyoe of data label')
    # Data settings
    parser.add_argument("--dataset", dest="dataset", default='cifar10', type=str,
                        choices=['mnist', 'cifar10', 'clothing1m'],
                        help="model input image size")
    parser.add_argument("--image_size", dest="image_size", default=224, type=int,
                        help="model input image size")
    parser.add_argument('--classnum', default=10, type=int,
                        metavar='H-P', help='number of train dataset classes')
    parser.add_argument('--device', dest='device', default=device, type=str,
                        help='select gpu')
    parser.add_argument('--data_device', dest="data_device",
                        default=data_device, type=int,
                        help="Dataset loading device."
                             "  0: load all data path;"
                             "  1: load all data in RAM."
                             "Default choice is 1. "
                             "Please ensure your computer have enough capacity!")
    parser.add_argument('--dataRoot', dest='root', default=cifar10_root,
                        type=str, metavar='PATH', help='where is the dataset')
    parser.add_argument('--datanum', default=15000, type=int,
                        metavar='H-P', help='number of train dataset samples')
    # training parameters group 5
    parser.add_argument("--gamma", dest="gamma", default=0.6, type=float,
                        help="forget rate schelduler param, should be negative.")
    parser.add_argument("--mu", dest="mu", default=-1, type=float,
                        help="setting for measure net divergence")
    parser.add_argument("--xi", dest="xi", default=0.01, type=float,
                        help="setting for weighted net param divergence")
    # training parameters group 6
    parser.add_argument('--save-ckpt', dest='save_ckpt', default=False, action='store_true')
    parser.add_argument('--record-in-train', dest='record_in_train', default=False, action='store_true')
    parser.add_argument('--dir', dest='dir', default="experiment/test-debug", type=str,
                        metavar='PATH', help='save dir')
    parser.add_argument('--random-seed', dest='random_seed', default=None, type=int,
                        metavar='N', help='random seed, default None.')

    args = parser.parse_args()

    # Setting of different dataset
    if args.dataset == 'mnist':
        print("Training on mnist")
        # args.backbone = 'cnn'
        if args.root == cifar10_root:
            args.root = mnist_root
        args.batch_size = 128
        args.image_size = 28
        args.classnum = 10
        args.input_dim = 1
        args.linear_num = 144
        args.datanum = 60000
        args.lr = 0.001
        args.lr2 = 0.0001
    elif args.dataset == 'cifar10':
        print("Training on cifar10")
        # args.backbone = 'cnn'
        if args.root == cifar10_root:
            args.root = cifar10_root
        args.warmup = 0
        args.batch_size = 128
        args.image_size = 32
        args.classnum = 10
        args.input_dim = 3
        args.datanum = 50000
    elif args.dataset == 'clothing1m':
        if args.root == cifar10_root:
            args.root = clothing1m_root
        args.data_device = 0
        args.backbone = 'resnet18'
        args.image_size = 224
        args.classnum = 14
        args.input_dim = 3
        args.datanum = 1000000
        args.stage1 = 5
        args.stage2 = 10
        args.epochs = 15
        args.batch_size = 64
        args.dim_reduce = 256
        args.noise_type = 'clean'
    else:
        print("Use default setting")

    return args