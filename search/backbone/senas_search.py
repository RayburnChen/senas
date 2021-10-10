import torch
from torch.functional import F

from models import BasicBlock
from util.operations import *
from util.genotype import *
from search.backbone.cell import Cell


class Head(nn.Module):

    def __init__(self, meta_node_num, double_down, c_in0, c_in1, nclass):
        super(Head, self).__init__()
        self.up_cell = Cell(meta_node_num, double_down, c_in0, c_in1, c_in1, cell_type='up')
        self.segmentation_head = ReLUConv(c_in1, nclass, kernel_size=3)

    def forward(self, s0, ot, weights_up_norm, weights_up, betas_up):
        return self.segmentation_head(self.up_cell(s0, ot, weights_up_norm, weights_up, betas_up))


class SenasSearch(nn.Module):

    def __init__(self, in_channels, c, nclass, depth, meta_node_num=3, double_down_channel=False, supervision=False):
        super(SenasSearch, self).__init__()
        self._depth = depth
        self._double_down_channel = double_down_channel
        self._supervision = supervision
        self._meta_node_num = meta_node_num

        assert self._depth >= 2, 'depth must >= 2'
        double_down = 2 if self._double_down_channel else 1
        c_in0, c_in1, c_curr = c, c, c

        self.blocks = nn.ModuleList()
        self.stem0 = ConvBn(in_channels, c_in0, kernel_size=7)
        stem1_pool = nn.MaxPool2d(3, stride=2, padding=1)
        stem1_block = BasicBlock(c_in0, c_in1, stride=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d)
        self.stem1 = nn.Sequential(build_activation(False), stem1_pool, stem1_block)

        num_filters = []
        down_f = []
        down_block = nn.ModuleList()
        for i in range(self._depth):
            if i == 0:
                filters = [1, 1, int(c_in1), 'stem1']
                down_cell = self.stem1
                down_f.append(filters)
                down_block += [down_cell]
            else:
                c_curr = int(double_down * c_curr)
                filters = [c_in0, c_in1, c_curr, 'down']
                down_cell = Cell(meta_node_num, double_down, c_in0, c_in1, c_curr, cell_type='down')
                down_f.append(filters)
                down_block += [down_cell]
                c_in0, c_in1 = c_in1, c_curr

        num_filters.append(down_f)
        self.blocks += [down_block]

        for i in range(1, self._depth):
            up_f = []
            up_block = nn.ModuleList()
            for j in range(self._depth - i):
                _, _, head_curr, _ = num_filters[0][j]
                _, _, head_down, _ = num_filters[i - 1][j + 1]
                head_in0 = sum([num_filters[k][j][2] for k in range(i)])  # up_cell._multiplier
                head_in1 = head_down  # up_cell._multiplier
                filters = [head_in0, head_in1, head_curr, 'up']
                up_cell = Cell(meta_node_num, double_down, head_in0, head_in1, head_curr, cell_type='up')
                up_f.append(filters)
                up_block += [up_cell]
            num_filters.append(up_f)
            self.blocks += [up_block]

        self.head_block = nn.ModuleList()

        c_in0 = c
        c_in1 = num_filters[-1][0][2]
        self.head_block.append(Head(meta_node_num, double_down, c_in0, c_in1, nclass))

    def forward(self, x, alpha_dn_nm, alpha_up_nm, alpha_dn, alpha_up, beta_dn, beta_up, gamma):
        # alpha_dn_nm is params for down cell normal ops
        # alpha_up_nm is params for up cell normal ops
        # alpha_dn is params for down cell down ops
        # alpha_up is params for up cell up ops
        cell_out = []
        for j, cell in enumerate(self.blocks[0]):
            if j == 0:
                # stem0: 1x256x256 -> 32x256x256
                s0 = self.stem0(x)
                # stem1: 32x256x256 -> 32x128x128
                ot = cell(s0)
                cell_out.append(ot)
            elif j == 1:
                ot = cell(s0, cell_out[-1], alpha_dn_nm, alpha_dn, beta_dn)
                cell_out.append(ot)
            else:
                ot = cell(cell_out[-2], cell_out[-1], alpha_dn_nm, alpha_dn, beta_dn)
                cell_out.append(ot)

        for j in reversed(range(self._depth - 1)):
            for i in range(1, self._depth - j):
                ides = list(range(j, i + j))
                gamma_ides = [sum(range(k + j)) + j for k in range(1, i)]
                in0 = torch.cat(
                    [cell_out[ides[0]]] + [cell_out[ides[k]] * gamma[idx][0] + cell_out[ides[k + 1]] * gamma[idx][1] for
                                           k, idx in enumerate(gamma_ides)], dim=1)
                # in0 = torch.cat([cell_out[idx] for idx in ides], dim=1)
                in1 = cell_out[i + j]
                cell = self.blocks[i][j]
                ot = cell(in0, in1, alpha_up_nm, alpha_up, beta_up)
                cell_out[i + j] = ot

        if self._supervision:
            return [self.head_block[-1](s0, ot, alpha_up_nm, alpha_up, beta_up) for ot in cell_out]
        else:
            return [self.head_block[-1](s0, cell_out[-1], alpha_up_nm, alpha_up, beta_up)]


class NAS(nn.Module):

    def __init__(self, input_c, c, num_classes, depth, meta_node_num=4,
                 use_sharing=True, double_down_channel=True, use_softmax_head=False, supervision=False,
                 multi_gpus=False, device='cuda'):
        super(NAS, self).__init__()
        self._use_sharing = use_sharing
        self._meta_node_num = meta_node_num
        self._depth = depth

        self.net = SenasSearch(input_c, c, num_classes, self._depth, meta_node_num, double_down_channel, supervision)
        # init weight using hekming methods
        self.net.apply(weights_init)

        if 'cuda' == str(device.type) and multi_gpus:
            device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids
        else:
            self.device_ids = [0]

        # Initialize architecture parameters: alpha
        self._init_alphas()

    def _init_alphas(self):

        normal_num_ops = len(NormOps)
        down_num_ops = len(DownOps)
        up_num_ops = len(UpOps)

        k = sum(1 for i in range(self._meta_node_num) for n in range(2 + i))  # total number of input node
        self.alphas_dn = nn.Parameter(1e-3 * torch.randn(k, down_num_ops))
        self.alphas_up = nn.Parameter(1e-3 * torch.randn(k, up_num_ops))
        self.alphas_dn_nm = nn.Parameter(1e-3 * torch.randn(k, normal_num_ops))
        self.alphas_up_nm = self.alphas_dn_nm if self._use_sharing else nn.Parameter(
            1e-3 * torch.randn(k, normal_num_ops))

        self.betas_dn = nn.Parameter(1e-3 * torch.randn(k))
        self.betas_up = nn.Parameter(1e-3 * torch.randn(k))

        self.gamma = nn.Parameter(1e-3 * torch.randn(sum(range(self._depth - 1)), 2))

        # alpha_dn_nm is params for down cell normal ops
        # alpha_up_nm is params for up cell normal ops
        # alpha_dn is params for down cell down ops
        # alpha_up is params for up cell up ops
        self._arch_parameters = [
            self.alphas_dn,
            self.alphas_up,
            self.alphas_dn_nm,
            self.alphas_up_nm,
            self.betas_dn,
            self.betas_up,
            self.gamma
        ]

    def load_params(self, alphas_dict, betas_dict):
        self.alphas_dn = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_dn_nm = alphas_dict['alphas_normal_down']
        self.alphas_up_nm = alphas_dict['alphas_normal_up']
        self.betas_dn = betas_dict['betas_down']
        self.betas_up = betas_dict['betas_up']
        self._arch_parameters = [
            self.alphas_dn,
            self.alphas_up,
            self.alphas_dn_nm,
            self.alphas_up_nm,
            self.betas_dn,
            self.betas_up
        ]

    def alphas_dict(self):
        return {
            'alphas_dn': self.alphas_dn,
            'alphas_dn_nm': self.alphas_dn_nm,
            'alphas_up': self.alphas_up,
            'alphas_up_nm': self.alphas_up_nm,
        }

    def betas_dict(self):
        return {
            'betas_dn': self.betas_dn,
            'betas_up': self.betas_up
        }

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # Note: Since we stack cells by s0: prev prev cells output; s1: prev cells output
        # and when a cell is a up cell, the s0 will be horizontal input and can't do up operation
        # which is different from down cells (s0 and s1 all need down operation). so when
        # parse a up cell string, the string operations is |_|*|_|...|_|, where * indicate up operation
        # mask1 and mask2 below is convenient to handle it.
        alphas_dn_nm = F.softmax(self.alphas_dn_nm, dim=-1).detach().cpu()
        alphas_dn = F.softmax(self.alphas_dn, dim=-1).detach().cpu()
        alphas_up_nm = F.softmax(self.alphas_up_nm, dim=-1).detach().cpu()
        alphas_up = F.softmax(self.alphas_up, dim=-1).detach().cpu()
        betas_dn = []
        betas_up = []
        for i in range(self._meta_node_num):
            offset = len(betas_dn)
            betas_dn.append(F.softmax(self.betas_dn[offset:offset + 2 + i], dim=-1).detach().cpu())
            betas_up.append(F.softmax(self.betas_up[offset:offset + 2 + i], dim=-1).detach().cpu())
        betas_dn = torch.cat(betas_dn, dim=0)
        betas_up = torch.cat(betas_up, dim=0)

        k = sum(1 for i in range(self._meta_node_num) for n in range(2 + i))  # total number of input node
        for j in range(k):
            alphas_dn_nm[j, :] = alphas_dn_nm[j, :] * betas_dn[j].item()
            alphas_dn[j, :] = alphas_dn[j, :] * betas_dn[j].item()
            alphas_up_nm[j, :] = alphas_up_nm[j, :] * betas_up[j].item()
            alphas_up[j, :] = alphas_up[j, :] * betas_up[j].item()

        geno_parser = GenoParser(self._meta_node_num)
        gene_down = geno_parser.parse(alphas_dn_nm.numpy(), alphas_dn.numpy(), cell_type='down')
        gene_up = geno_parser.parse(alphas_up_nm.numpy(), alphas_up.numpy(), cell_type='up')
        concat = range(2, self._meta_node_num + 2)
        gamma = F.softmax(self.gamma, dim=-1).detach().cpu()
        idx = torch.topk(gamma[:, 1], len(gamma) // 2, largest=False).indices
        gamma = gamma.argmax(1).tolist()
        gamma = [g if i not in idx else 0 for i, g in enumerate(gamma)]
        gamma_path = [gamma[sum(range(i)): sum(range(i)) + i] for i in range(1, self._depth - 1)]
        gamma_path = sum([g[:g.index(1)] + [1] * len(g[g.index(1):]) if i in g else g for g in gamma_path], [])
        geno_type = Genotype(
            down=gene_down, down_concat=concat,
            up=gene_up, up_concat=concat,
            gamma=gamma_path
        )
        return geno_type

    def forward(self, x):

        alphas_dn_nm = F.softmax(self.alphas_dn_nm, dim=-1)
        alphas_up_nm = F.softmax(self.alphas_up_nm, dim=-1)
        alphas_dn = F.softmax(self.alphas_dn, dim=-1)
        alphas_up = F.softmax(self.alphas_up, dim=-1)
        betas_dn = []
        betas_up = []
        for i in range(self._meta_node_num):
            offset = len(betas_dn)
            betas_dn.append(F.softmax(self.betas_dn[offset:offset + 2 + i], dim=-1))
            betas_up.append(F.softmax(self.betas_up[offset:offset + 2 + i], dim=-1))
        betas_dn = torch.cat(betas_dn, dim=0)
        betas_up = torch.cat(betas_up, dim=0)
        gamma = F.softmax(self.gamma, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, alphas_dn_nm, alphas_up_nm, alphas_dn, alphas_up, betas_dn, betas_up, gamma)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        w_dn_nm_copies = broadcast_list(alphas_dn_nm, self.device_ids)
        w_up_nm_copies = broadcast_list(alphas_up_nm, self.device_ids)
        w_dn_copies = broadcast_list(alphas_dn, self.device_ids)
        w_up_copies = broadcast_list(alphas_up, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs, w_dn_nm_copies, w_up_nm_copies,
                                                                w_dn_copies, w_up_copies)),
                                             devices=self.device_ids)

        return nn.parallel.gather(outputs, self.device_ids[0])


class Architecture(object):

    def __init__(self, model, arch_optimizer, criterion):
        self.model = model
        self.optimizer = arch_optimizer
        self.criterion = criterion

    def step(self, input_valid, target_valid):
        """Do one step of gradient descent for architecture parameters

        Args:
            input_valid: A tensor with N * C * H * W for validation data
            target_valid: A tensor with N * 1 for validation target
            eta:
            network_optimizer:
        """

        self.optimizer.zero_grad()
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()
        self.optimizer.step()
