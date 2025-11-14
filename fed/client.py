import torchvision.models as models
from fed.aggregation import FedProximal, quantize_client
from utils.utils import *
from utils.datasets import *
from utils.models import *

def compute_loss_wm(rebuild_wms, vectors, key, wm, device, ssim_threshold=0.95, m=0.1,
                 data_range=1, y=0.5):
    rebuild_wms = rebuild_wms.to(device)
    vectors = vectors.to(device)
    key = key.to(device)
    wm = wm.to(device)

    sims = torch.cosine_similarity(vectors, key, dim=1)
    p_sim_mask = sims >= ssim_threshold
    n_sim_mask = ~p_sim_mask

    ssim_p = torch.zeros_like(sims, device=device)
    ssim_n = torch.zeros_like(sims, device=device)
    l1_p = torch.zeros_like(sims, device=device)

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

    if n_sim_mask.any():
        ssim_n[n_sim_mask] = torch.relu(
            torch.abs(
                ssim(
                    rebuild_wms[n_sim_mask],
                    wm.expand(n_sim_mask.sum(), -1, -1, -1),
                    data_range=data_range
                )) - m
        )

    loss = (y * (1 - ssim_p) + (1-y) * ssim_n).mean()
    loss = loss + (1.0 * l1_p).mean()

    return loss

class Client:
    def __init__(self):
        self.model = None
        self.T_model = None
        self.dataset = None

    def set_model(self, args):
        self.model = get_model(args).to(args.device)

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
            self.VecDataset = VectorAugment(wm_dataset, wm_idx, self.dim, self.args.num)
            self.key_vec = self.VecDataset.key_vec
            self.wm = self.VecDataset.wm
            self.seed = self.VecDataset.seed
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
                                p.grad.data -= lc - gc

                    optimizer.step()

                    batch_loss.append(loss.item())

                epoch_loss.append(np.mean(batch_loss))

            test_acc, test_ave_loss = evaluate(self.model, self.args)
            print(f"client: {self.order}, num_data: {len(self.local_dataset)}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}")

        else :
            Vec_Loader = DataLoader(self.VecDataset, batch_size=self.wm_bs, shuffle=True)
            iter_vec = iter(Vec_Loader)

            self.model.to(self.device)
            self.T_model.to(self.device)

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

                    rebuild_wms = self.T_model(vectors)

                    loss_wm = compute_loss_wm(
                                rebuild_wms=rebuild_wms,
                                vectors=vectors,
                                key=self.key_vec,
                                wm=self.wm,
                                device=self.device,
                                ssim_threshold=0.95,
                                m=self.args.m,
                                data_range=1,
                                y = self.args.y
                                )
                    lambda_wm = self.args.la
                    loss = loss_m + lambda_wm * loss_wm

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

            for name, module in self.T_model.named_modules():
                if isinstance(module, TransposedBatchNorm):
                    self.bn_stats[name] = {
                        'running_mean': module.running_mean.clone(),
                        'running_var': module.running_var.clone()
                    }

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