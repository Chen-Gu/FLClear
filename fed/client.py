
import torchvision.models as models

from fed.aggregation import FedProximal, quantize_client
from utils.utils import *
from utils.datasets import *
from utils.models import *

def get_loss_weights(total_rounds, current_round):
    # Start with a low watermark weight (0.2) and increase to 0.6 gradually
    initial_weight = 0.95
    target_weight = 0.4
    if current_round <= total_rounds / 5:
        weight = initial_weight
    elif current_round < (2 * total_rounds / 5):
        # Linear increase from initial to target over the middle third
        progress = (current_round - total_rounds / 5) / (total_rounds / 5)
        weight = initial_weight - progress * (target_weight - initial_weight)
    else:
        weight = target_weight
    classification_weight = 1.0 - weight
    return classification_weight, weight
def compute_gradnorm_weights(loss_m, loss_con, model, alpha=1.5):
    # 保存初始损失值，用于相对损失比例
    loss_m_init = loss_m.item()
    loss_con_init = loss_con.item()

    # 计算任务梯度
    model.zero_grad()
    loss_m.backward(retain_graph=True)
    grad_m = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    grad_norm_m = torch.norm(grad_m, p=2)

    model.zero_grad()
    loss_con.backward(retain_graph=True)
    grad_con = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    grad_norm_con = torch.norm(grad_con, p=2)

    # 计算目标梯度范数（平均值）
    avg_grad_norm = (grad_norm_m + grad_norm_con) / 2

    # 计算相对损失比例
    total_loss = loss_m_init + loss_con_init + 1e-8  # 防止除零
    loss_ratio_m = loss_m_init / total_loss
    loss_ratio_con = loss_con_init / total_loss

    # 计算 GradNorm 权重
    T = 1.0  # 初始任务权重
    alpha_m = T * (avg_grad_norm / (grad_norm_m + 1e-8)).pow(alpha) * loss_ratio_m
    alpha_con = T * (avg_grad_norm / (grad_norm_con + 1e-8)).pow(alpha) * loss_ratio_con

    # 归一化权重
    total_alpha = alpha_m + alpha_con + 1e-8
    alpha_m = alpha_m / total_alpha
    alpha_con = alpha_con / total_alpha

    return alpha_m, alpha_con
def pcgrad_gradients(grads_list, model_params):
    """
    grads_list: 每个任务的梯度列表，[[param1_grad, param2_grad, ...], [param1_grad, param2_grad, ...]]
    model_params: 模型参数生成器（如 model.parameters()）
    返回: 修正后的梯度列表
    """
    num_tasks = len(grads_list)
    modified_grads = [list(g) for g in grads_list]
    model_params_list = list(model_params)  # 转换为列表

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                dot_product = sum(
                    torch.sum(g_i * g_j)
                    for g_i, g_j in zip(grads_list[i], grads_list[j])
                    if g_i is not None and g_j is not None
                )
                if dot_product < 0:
                    norm_j = sum(
                        torch.sum(g_j * g_j)
                        for g_j in grads_list[j]
                        if g_j is not None
                    )
                    for k, (g_i, g_j) in enumerate(zip(grads_list[i], grads_list[j])):
                        if g_i is not None and g_j is not None:
                            modified_grads[i][k] = g_i - (dot_product / norm_j) * g_j

    final_grads = []
    for param_idx in range(len(model_params_list)):  # 使用列表
        grad_sum = None
        for task_grads in modified_grads:
            if task_grads[param_idx] is not None:
                if grad_sum is None:
                    grad_sum = task_grads[param_idx].clone()
                else:
                    grad_sum += task_grads[param_idx]
        final_grads.append(grad_sum)
    return final_grads


def compute_loss_wm(rebuild_wms, vectors, key, wm, device, ssim_threshold=1.0, ssim_low_target=0.1,
                 data_range=1):

    rebuild_wms = rebuild_wms.to(device)
    vectors = vectors.to(device)
    key = key.to(device)
    wm = wm.to(device)

    # 计算余弦相似度
    sims = torch.cosine_similarity(vectors, key, dim=1)  # 形状: (batch_size,)
    p_sim_mask = sims >= ssim_threshold  # 正样本掩码
    n_sim_mask = ~p_sim_mask  # 负样本掩码

    # 初始化损失张量
    ssim_p = torch.zeros_like(sims, device=device)
    ssim_n = torch.zeros_like(sims, device=device)
    l1_p = torch.zeros_like(sims, device=device)

    # 正样本损失（高相似度）
    if p_sim_mask.any():
        ssim_p[p_sim_mask] = ssim(
            rebuild_wms[p_sim_mask],
            wm.expand(p_sim_mask.sum(), -1, -1, -1),
            data_range=data_range
        )
        l1_p[p_sim_mask] = F.l1_loss(
            rebuild_wms[p_sim_mask],
            wm.expand(p_sim_mask.sum(), -1, -1, -1),
            reduction='none'
        ).mean(dim=(1, 2, 3))

    # 负样本损失（低相似度）
    if n_sim_mask.any():
        ssim_n[n_sim_mask] = torch.relu(
            torch.abs(
                ssim(
                    rebuild_wms[n_sim_mask],
                    wm.expand(n_sim_mask.sum(), -1, -1, -1),
                    data_range=data_range
                )) - ssim_low_target
        )
    loss = (0.01 * (1 - ssim_p) + 0.99 * ssim_n).mean()
    loss = loss + (1.0 * l1_p).mean()

    return loss

class Client:
    def __init__(self):
        self.model = None
        self.T_model = None
        self.dataset = None
    def set_model(self, args):
        self.model = get_model(args).to(args.device)
        # self.model = model

    def set_dataset(self, dataset):
        self.dataset = dataset
    def set_T_model(self, model):
        pass

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def get_T_local_model(self):
        return self.T_model

    def train_one_iteration(self, iter_num, num):
        pass
class OrdinaryClient(Client):
    def __init__(self, args, dataset=None, idx=None, wm_dataset=None, wm_idx=None):
        super().__init__()
        self.args = args
        self.order = 0
        self.CrossEntropy = torch.nn.CrossEntropyLoss()
        self.device = args.device
        self.optim = args.local_optim
        self.local_lr = args.local_lr
        self.momentum = args.local_momentum
        self.local_dataset = DatasetSplit(dataset, idx)
        self.dataLoader = DataLoader(self.local_dataset, batch_size=args.local_bs, shuffle=True)
        self.local_ep = args.local_ep
        self.agg = args.aggregation
        self.decay = args.lr_decay
        if wm_dataset is not None:
            self.wm_bs = args.wm_bs
            self.bn_stats = {}
            self.dim = args.num_classes
            self.VecDataset = VectorAugment(wm_dataset, wm_idx, self.dim)
            self.key_vec = self.VecDataset.key_vec
            self.wm = self.VecDataset.wm
            self.seed = self.VecDataset.seed  #torch.Size([1, 10]) torch.Size([1, 3, 32, 32])
            self.wm = self.wm.unsqueeze(0).to(dtype=torch.float32).to(self.device)
            self.key_vec = self.key_vec.unsqueeze(0).to(dtype=torch.float32).to(self.device)
        if self.agg == 'scaffold':
            self.local_c = None
            self.is_set_local_c = False
    def set_T_model(self, model):
        self.T_model = create_T_model(model, self.args.model)

    def local_update(self, global_model = None, global_c = None):
        torch.set_printoptions(precision=4, sci_mode=False)

        if self.agg == 'scaffold' and self.is_set_local_c == False:
            self.local_c = [torch.zeros_like(p.data) for p in self.model.parameters()]
            self.is_set_local_c = True

        epoch_loss = []
        optimizer = get_optim(self.model, self.optim, self.local_lr, self.momentum, self.decay)

        if not self.args.watermark:

            self.model.to(self.device)
            self.model.train()

            for epoch in range(self.args.local_ep):
                batch_loss = []

                for images, labels in self.dataLoader:

                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.CrossEntropy(outputs, labels)

                    if self.agg == 'FedProx' and global_model is not None:
                        prox_loss = FedProximal(self.model, global_model, mu=0.01)
                        loss = loss + prox_loss

                    optimizer.zero_grad()
                    loss.backward()

                    if self.agg == 'scaffold' and global_c is not None and self.local_c is not None:
                        with torch.no_grad():
                            for p, gc, lc in zip(self.model.parameters(), global_c, self.local_c):
                                # if p.grad is not None:
                                p.grad.data -= lc - gc

                    optimizer.step()

                    batch_loss.append(loss.item())

                epoch_loss.append(np.mean(batch_loss))

            test_acc, test_ave_loss = evaluate(self.model, self.args)
            print(f"client: {self.order}, num_data: {len(self.local_dataset)}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}")

        else :
            # ------------------------------------param set----------------------

            Vec_Loader = DataLoader(self.VecDataset, batch_size=self.wm_bs, shuffle=True)
            iter_vec = iter(Vec_Loader)

            self.model.to(self.device)
            self.T_model.to(self.device)

            # ------------------------------ Joint Training ----------------------
            for epoch in range(self.local_ep):

                self.T_model.train()
                self.model.train()

                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.dataLoader):

                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs_m = self.model(images)
                    loss_m = self.CrossEntropy(outputs_m, labels)

                    try:
                        vectors = next(iter_vec)
                        vectors = vectors.to(self.device)
                    except StopIteration:
                        iter_vec = iter(Vec_Loader)
                        vectors = next(iter_vec)
                        vectors = vectors.to(self.device)
                    # print(self.key_vec.shape, self.wm.shape)
                    # rebuild_wm = self.T_model(self.key_vec)
                    # loss_wm = 1 - ssim(rebuild_wm, self.wm, data_range=1)
                    rebuild_wms = self.T_model(vectors)
                    loss_wm = compute_loss_wm(
                                rebuild_wms=rebuild_wms,
                                vectors=vectors,
                                key=self.key_vec,
                                wm=self.wm,
                                device=self.device,
                                ssim_threshold=0.95,
                                ssim_low_target=0.1,
                                data_range=1
                                )


                    # lambda_m, lambda_wm = get_loss_weights(100, iter_num)
                    # lambda_m, lambda_wm = compute_gradnorm_weights(loss_m,loss_wm,self.model)
                    lambda_m, lambda_wm = 1.0, 1.5
                    loss = lambda_m * loss_m + lambda_wm * loss_wm

                    if self.agg == 'FedProx' :
                        prox_loss = FedProximal(self.model, global_model, mu=0.01)
                        loss = loss + prox_loss



                    batch_loss.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()

                    if self.agg == 'scaffold' and global_c is not None and self.local_c is not None:
                        with torch.no_grad():
                            for p, gc, lc in zip(self.model.parameters(), global_c, self.local_c):
                                if p.grad is not None:
                                    p.grad -= lc - gc

                    optimizer.step()

                epoch_loss.append(np.mean(batch_loss))


            # ------------------------------train over!----------------------
            for name, module in self.T_model.named_modules():
                if isinstance(module, TransposedBatchNorm):
                    self.bn_stats[name] = {
                        'running_mean': module.running_mean.clone(),
                        'running_var': module.running_var.clone()
                    }

            self.model.eval()
            self.T_model.eval()
            with torch.no_grad():
                rebuild_wm = self.T_model(self.key_vec)
                ssim_v = ssim(rebuild_wm, self.wm, data_range=1)
                test_acc, test_ave_loss = evaluate(self.model, self.args)
                print(f"client: {self.order}, num_data: {len(self.local_dataset)}, joint training: acc: {test_acc}%, SSIM: {ssim_v.item():.4f}")
                # print(f"lambda_m: {lambda_m:.2f}, lambda_wm: {lambda_wm:.2f}")
        self.model.train()

        model_state = self.model.state_dict()
        if self.agg == 'FedPAQ' and global_model is not None:
            model_state = quantize_client(self.model.state_dict(), global_model.state_dict(), quantization_bits = 8)
        if self.agg == 'scaffold' and global_model is not None:
            delta_c = []
            new_c = []
            with torch.no_grad():
                for gp, lp, gc, lc in zip(global_model.parameters(), self.model.parameters(), global_c, self.local_c):
                    delta = (gp.data - lp.data) / (self.local_ep * len(self.dataLoader))
                    c_new = lc - gc + delta
                    delta_c.append(c_new - lc)
                    new_c.append(c_new)
            self.local_c = new_c
            return model_state, len(self.local_dataset), np.mean(epoch_loss), delta_c
        return model_state, len(self.local_dataset), np.mean(epoch_loss)

def create_clients(args, dataset, wm_dataset=None):

    if args.distribution == 'iid':
        idxes = iid_split(dataset, args.num_clients)
    elif args.distribution == 'dnon-iid':
        idxes = dniid_split(dataset, args.num_clients, args.dniid_param)

    elif args.distribution == 'pnon-iid':
        idxes = pniid_split(dataset, args.num_clients)
    else:
        exit("Unknown Distribution!")

    wm_idxes = dict()
    if wm_dataset is not None:
         wm_idxes = iid_split(wm_dataset, args.num_clients)
    clients = []

    for i, idx in enumerate(idxes.values()):
        if wm_dataset is not None:
            client = OrdinaryClient(args, dataset, idx, wm_dataset,  wm_idxes[i])
        else:
            client = OrdinaryClient(args, dataset, idx, None, None)
        clients.append(client)
    return clients