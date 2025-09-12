# -*- coding: UTF-8 -*-

import copy
import os.path
import random
import time

import matplotlib.pyplot as plt
from torch.backends import cudnn
from fed.client import create_clients
from fed.aggregation import FedAvg, FedMax, ties_merging, FedAvgWithTrimming, fedadam_aggregate
from utils.datasets import WMDataset
from utils.parameters import load_args
from utils.utils import *
from tqdm import tqdm


def compare_state_dicts(state_dict1, state_dict2):
    """
    比较两个模型的state_dict是否完全相同

    Args:
        model1: 第一个PyTorch模型
        model2: 第二个PyTorch模型

    Returns:
        bool: 如果state_dict完全相同返回True，否则返回False
    """
    # state_dict1 = model1.state_dict()
    # state_dict2 = model2.state_dict()

    # 检查是否有相同的键
    if state_dict1.keys() != state_dict2.keys():
        return False

    # 比较每个键对应的张量
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

def main(args):

    '''set seed，确保实验可再现'''
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    '''---------------------------load dataset-------------------------------'''
    # print(f"-----------{args.dataset}loading--------------")
    train_dataset = get_full_dataset(args.dataset, train=True, img_size=(32, 32), is_download=False)
    wmDataset = WMDataset(image_dir=args.wm_dir, dim = args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))

    global_model = get_model(args).to(args.device)
    if args.watermark:
        global_T_model = create_T_model(global_model, args.model).to(args.device)

    ''' 创建客户端 , 包括[数据集，训练包装函数，训练参数，待创建模型] '''
    if args.watermark :
        clients = create_clients(args, train_dataset, wmDataset)
    else:
        clients = create_clients(args, train_dataset, None)

    ''' 为每个客户端创建模型 '''
    for num, client in enumerate(clients):
        client.order = num + 1
        client.set_model(args)
        # client.set_model(copy.deepcopy(global_model))
        if args.watermark:
            client.set_T_model(client.model)

    ''' client：获取新的模型参数 '''
    for client in clients:
        client.model.load_state_dict(global_model.state_dict())

    ''' client: 客户端迭代训练 '''
    train_loss = []
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)
    best_acc = 0

    if args.aggregation == 'scaffold':
        global_c = [torch.zeros_like(p.data) for p in global_model.parameters()]
    if args.aggregation == 'FedAdam':
        server_state = {}

    with open(args.save_dir + args.save_file + '.txt', 'w') as file:
        for epoch in tqdm(range(args.start_epochs, args.epochs)):
            print("\n")
            start_time = time.time()
            local_losses = []
            local_models = []
            local_nums = []
            orders = []
            delta_cs = []

            '''调整学习率并选择参与该轮训练的客户端'''
            for client in clients:
                client.local_lr -= client.local_lr * args.lr_decay
            clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False)

            '''被选中的客户端进行本地训练'''
            for num, idx in enumerate(clients_idxs):
                current_client = clients[idx]
                orders.append(current_client.order)

                if args.aggregation == 'scaffold':
                    local_model, num_samples, local_loss, delta_c = current_client.local_update(
                        global_model=global_model,
                        global_c=global_c
                    )
                    delta_cs.append((delta_c, num_samples))
                elif args.aggregation == 'FedProx' or args.aggregation == 'FedPAQ':
                    local_model, num_samples, local_loss = current_client.local_update(
                        global_model=global_model
                    )
                else:
                    local_model, num_samples, local_loss= current_client.local_update()
                local_models.append(local_model)
                # local_models.append(copy.deepcopy(local_model))
                local_losses.append(local_loss)
                local_nums.append(num_samples)

            '''打印出每轮的损失值'''
            avg_loss = np.mean(local_losses)
            print(f"Round {epoch}, Average loss {avg_loss:.3f}, Time: {time.time() - start_time}")

            ''' server: 聚合模型更新 '''
            if args.aggregation == 'FedAdam':
                avg_update, server_state = fedadam_aggregate(global_model.state_dict(), local_models, local_nums, server_state, eta=args.fedAdam_lr)
            else:
                avg_update = FedAvg(local_models, local_nums)


            if args.aggregation == 'scaffold':
                total_samples = sum(local_nums)
                for i in range(len(global_c)):
                    delta_sum = sum(delta_c[i] * num for delta_c, num in delta_cs)
                    # if torch.isnan(delta_sum).any():
                    #     print(f"NaN detected in delta_c[{i}] at round {epoch}")
                    global_c[i] += delta_sum / total_samples

            global_model.load_state_dict(avg_update)

            ''' client：获取新的模型参数 '''
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())

            ''' 评估全局模型的准确率 '''
            test_acc, test_ave_loss = evaluate(global_model, args)
            if test_acc >= best_acc:
                best_acc = test_acc
                avg_update_save = avg_update
            '''水印提取'''
            if args.watermark:
                key_vecs = []
                wms = []
                ssims = []
                # ssim_vlaue = []
                build_wms = []

                print(f"----------------第{epoch+1}轮水印提取---------------------")
                global_T_model.eval()
                with torch.no_grad():
                    for num, client in enumerate(clients):
                        key_vecs.append(client.key_vec)
                        wms.append(client.wm)
                        if client.bn_stats != {}:
                            build_wm = global_T_model(client.key_vec, client.bn_stats)
                            ssim_v = ssim(build_wm, client.wm)
                            build_wms.append(build_wm)
                            ssims.append(ssim_v.item())
                    wms = torch.cat(wms, dim=0)
                    build_wms = torch.cat(build_wms, dim=0)
                    avg_ssim = np.mean(ssims)
                    if epoch == args.epochs - 1 :
                        plt.figure(figsize=(10, 4))
                        for i in range(10):
                            # 原始图像
                            plt.subplot(2, 10, i + 1)
                            imshow(wms[i])
                            if i == 0:
                                    plt.title(f"Original")
                            # 重建图像
                            plt.subplot(2, 10, i + 11)
                            if i < build_wms.shape[0]:
                                imshow(build_wms[i])
                            if i == 4:
                                plt.title(f"epoch Reconstructed result: {str(epoch + 1)}" )
                        plt.tight_layout()
                        plt.savefig(f"./result/AlexNet/img/{args.save_file}_ssim_{avg_ssim:0.2f}.png")
                        plt.close()

                    print(f"clients: {orders}")
                    print(f"agg epoch:{epoch}, test_accuracy: {test_acc:.2f}%, avg_ssim:{avg_ssim:.2f}, test_loss: {test_ave_loss:.2f}")
                    log_entry = f"Epoch {epoch}, avg_ssim:{avg_ssim:.2f}, test_accuracy: {test_acc:.2f}%, test_loss: {test_ave_loss:.2f}\n"
                    file.write(log_entry)

                    '''随机输出'''
                    if epoch == args.epochs - 1 :

                        if args.save :
                            torch.save(avg_update_save, args.save_model_dir + args.save_file + ".pth")
                            save_bn_stats_hdf5(clients, save_path=args.save_bn_dir + args.save_file + ".h5")
                            print(f"{args.model} has been saved! best_acc:{best_acc:.2f}")

                        vector_rand = RandVec_generate(args.num_classes, 36, args.device)
                        for num, client in enumerate(clients):
                            if client.bn_stats != {}:
                                # print("hhh")
                                rand_bn_out = global_T_model(vector_rand, client.bn_stats)
                                rand_out = global_T_model(vector_rand)
                                plt.figure(figsize=(8, 8))
                                for i in range(36):
                                    plt.subplot(6, 6, i + 1)
                                    imshow(rand_bn_out[i])
                                    if i == 4:
                                        plt.title(f"bn client seed: {client.seed}")
                                plt.tight_layout()
                                plt.savefig(f"./result/AlexNet/img/{args.save_file}_bn_{client.seed}.png", dpi=300)
                                plt.close()
                                plt.figure(figsize=(8, 8))
                                for i in range(36):
                                    plt.subplot(6, 6, i + 1)
                                    imshow(rand_out[i])
                                    if i == 4:
                                        plt.title(f"no_bn client seed: {client.seed}")
                                plt.tight_layout()
                                plt.savefig(f"./result/AlexNet/img/{args.save_file}_no_bn_{client.seed}.png", dpi=300)
                                plt.close()
                            break
                        # plt.show()

            else:

                print(f"agg epoch:{epoch}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}")
                log_entry = f"Epoch {epoch}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}\n"
                file.write(log_entry)


if __name__ == "__main__":
    args = load_args()
    main(args)
    # eta_list = [0.0001, 0.00001, 0.00005]
    # lr_list = [0.0001, 0.001]
    # if args.model == "VGG13":
    #     args.save_dir = "./result/VGG/"
    # elif args.model == "ResNet18":
    #     args.save_dir = "./result/ResNet/"
    # else:
    #     args.save_dir = "./result/AlexNet/"
    #
    #
    # for adam_lr in eta_list:
    #     for lr in lr_list:
    #         args.fedAdam_lr = adam_lr
    #         args.local_lr = lr
    #
    #         if args.watermark == True:
    #             args.save_file = "C10_Adam_" + str(adam_lr)+ "_" +str(args.local_lr) + "_wm"
    #         else:
    #             args.save_file = "C10_Adam_" + str(adam_lr)+ "_base"
    #         main(args)



