# -*- coding: UTF-8 -*-
import argparse
import distutils.util


def load_args():
    parser = argparse.ArgumentParser()
    # global settings
    parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
    parser.add_argument('--epochs', type=int, default=80, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients: K")
    parser.add_argument('--clients_percent', type=float, default=0.4, help="the fraction of clients to train the local models in each iteration.")

    # settings
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")
    parser.add_argument('--local_optim', type=str, default='sgd', help="local optimizer: [sgd, adam]")
    parser.add_argument('--local_lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--local_momentum', type=float, default=0.90, help="SGD momentum (default: 0)")
    parser.add_argument('--local_loss', type=str, default="CE", help="Loss Function")
    parser.add_argument('--distribution', type=str, default='dnon-iid', help="the distribution used to split the dataset")
    parser.add_argument('--dniid_param', type=float, default=0.8)
    parser.add_argument('--lr_decay', type=float, default=0.0005)

    parser.add_argument('--model', type=str, default='AlexNet', help='model name: [VGG13, AlexNet, ResNet18]')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset: [mnist, cifar10, cifar100]")
    parser.add_argument('--aggregation', type=str, default='FedAvg', help='agg name: [FedAvg, FedProx, FedPAQ, scaffold, FedAdam]')
    parser.add_argument('--fedAdam_lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--image_size', type=int, default=32, help="length or width of images")
    parser.add_argument('--device', type=str, default="cuda:6", help='Set the device to use: "cuda" or "cpu".')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default="./result/AlexNet/")
    parser.add_argument('--save_model_dir', type=str, default="./result/AlexNet/model/")
    parser.add_argument('--save_bn_dir', type=str, default="./result/AlexNet/bn/")
    parser.add_argument('--wm_dir', type=str, default="./data/logo10/")
    parser.add_argument('--save_file', type=str, default="VGG_C20_FedAvg_wm")
    parser.add_argument('--save', type=lambda x: bool(distutils.util.strtobool(x)), default=True)

    # # watermark arguments
    parser.add_argument("--watermark", type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether embedding the watermark")
    parser.add_argument('--wm_bs', type=int, default=32, help="watermark images batch size")
    # parser.add_argument('--wm_ep', type=int, default=2, help="the number of watermark training epochs")

    parser.add_argument('--num_trigger_set', type=int, default=10, help="the number of trigger watermark set")
    # parser.add_argument('--logdir', type=str, default='./log/temp/')
    args = parser.parse_args()
    args.num_clients_each_iter = int(args.num_clients * args.clients_percent)
    return args