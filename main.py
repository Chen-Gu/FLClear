# -*- coding: UTF-8 -*-

import copy
import os.path
import random
import time
import matplotlib.pyplot as plt
from torch.backends import cudnn
from fed.client import create_clients
from fed.aggregation import FedAvg, fedadam_aggregate
from utils.datasets import WMDataset
from utils.parameters import load_args
from utils.utils import *
from tqdm import tqdm


def main(args):

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    '''---------------------------load dataset-------------------------------'''
    train_dataset = get_full_dataset(args.dataset, train=True, img_size=(args.image_size, args.image_size), is_download=False)
    wmDataset = WMDataset(image_dir=args.wm_dir, dim = args.num_classes, transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                ]))
  
    global_model = get_model(args).to(args.device)
    if args.watermark:
        global_T_model = create_T_model(global_model, args.model).to(args.device)

    if args.watermark :
        clients = create_clients(args, train_dataset, wmDataset)
    else:
        clients = create_clients(args, train_dataset, None)

    for num, client in enumerate(clients):
        client.order = num + 1
        client.set_model(args)
        if args.watermark:
            client.set_T_model(client.model)

    for client in clients:
        client.model.load_state_dict(global_model.state_dict())

    train_loss = []
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)
    best_acc = 0
    best_ssim = 0

    if args.aggregation == 'scaffold':
        global_c = [torch.zeros_like(p.data) for p in global_model.parameters()]
    if args.aggregation == 'FedAdam':
        server_state = {}
    save_file = args.save_file 
    with open(args.save_dir + save_file + '.txt', 'w') as file:
        for epoch in tqdm(range(args.start_epochs, args.epochs)):
            print("\n")
            start_time = time.time()
            local_losses = []
            local_models = []
            local_nums = []
            orders = []
            delta_cs = []

            for client in clients:
                client.local_lr -= client.local_lr * args.lr_decay
            clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False)

    
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
                local_losses.append(local_loss)
                local_nums.append(num_samples)

            avg_loss = np.mean(local_losses)
            print(f"Round {epoch}, Average loss {avg_loss:.3f}, Time: {time.time() - start_time}")

            if args.aggregation == 'FedAdam':
                avg_update, server_state = fedadam_aggregate(global_model.state_dict(), local_models, local_nums, server_state, eta=args.fedAdam_lr)
            else:
                avg_update = FedAvg(local_models, local_nums)

            if args.aggregation == 'scaffold':
                total_samples = sum(local_nums)
                for i in range(len(global_c)):
                    delta_sum = sum(delta_c[i] * num for delta_c, num in delta_cs)
                    global_c[i] += delta_sum / total_samples

            global_model.load_state_dict(avg_update)

           
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())

           
            test_acc, test_ave_loss = evaluate(global_model, args)
                     
            if args.watermark:
                key_vecs = []
                wms = []
                ssims = []
                mse_v = []
                build_wms = []

                global_T_model.eval()
                with torch.no_grad():
                    for num, client in enumerate(clients):
                        key_vecs.append(client.key_vec)
                        wms.append(client.wm)
                        if client.bn_stats != {}:
                         
                            build_wm = global_T_model(client.key_vec, client.bn_stats)
                            ssim_v = ssim(build_wm, client.wm)
                            mse = F.mse_loss(build_wm, client.wm)
                            build_wms.append(build_wm)
                            ssims.append(ssim_v.item())
                            mse_v.append(mse.item())

                    avg_ssim = np.mean(ssims)
                    avg_mse = np.mean(mse_v)
                                 
                    print(f"agg epoch:{epoch}, test_accuracy: {test_acc:.2f}%, avg_ssim:{avg_ssim:.2f}, test_loss: {test_ave_loss:.2f}, avg_mse:{avg_mse:.2f}")
                    log_entry = f"Epoch {epoch}, avg_ssim:{avg_ssim:.2f}, test_accuracy: {test_acc:.2f}%, test_loss: {test_ave_loss:.2f}, avg_mse:{avg_mse:.2f}\n"
                    file.write(log_entry)
             
                    if epoch == args.epochs - 1 :
                        if args.save :
                            torch.save(avg_update, args.save_model_dir + save_file + ".pth")
                            save_bn_stats_hdf5(clients, save_path=args.save_bn_dir + save_file + ".h5")
                            print(f"{args.model} has been saved! acc:{best_acc:.2f}, ssim: {best_ssim:.2f}")                     
            else:

                print(f"agg epoch:{epoch}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}")
                log_entry = f"Epoch {epoch}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}\n"
                file.write(log_entry)
                if epoch == args.epochs - 1:
                    if args.save:
                        torch.save(avg_update, args.save_model_dir + save_file + ".pth")
                        print(f"{args.model} has been saved!")


if __name__ == "__main__":
    args = load_args()
    main(args)




