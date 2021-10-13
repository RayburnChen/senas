from torch.functional import F
from utils.operations import *


class MixedOp(nn.Module):

    def __init__(self, c_in, c_out, op_type):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_type = op_type
        # note:
        # this k is a similar parameter as PC-DARTS
        # the k parameter of Senas is in the Cell class
        self.k = 1
        self.c_out = c_out
        self.c_part = int(c_out // self.k)

        if self.c_out - self.c_part > 0:
            if OpType.DOWN == self._op_type:
                # down cell needs pooling before concat
                self.skip = nn.MaxPool2d(2, stride=2)
            elif OpType.UP == self._op_type:
                # up cell needs interpolate before concat
                self.skip = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            else:
                self.skip = nn.Identity()

        for pri in self._op_type.value['ops']:
            op = OPS[pri](c_in, self.c_part, self._op_type, dp=0)
            self._ops.append(op)

    def forward(self, x, alpha_normal, alpha_up_dn):
        if OpType.NORM == self._op_type:
            out = sum(w * op(x) for w, op in zip(alpha_normal, self._ops))
        else:
            out = sum(w * op(x) for w, op in zip(alpha_up_dn, self._ops))

        if self.c_out - self.c_part > 0:
            # > 1/k
            x_part_2 = x[:, -self.c_out + self.c_part:, :, :]
            out = torch.cat([out, self.skip(x_part_2)], dim=1)
            out = channel_shuffle(out, self.k)
        return out


class Cell(nn.Module):

    def __init__(self, meta_node_num, double_down, c_in0, c_in1, c_out, cell_type):
        super(Cell, self).__init__()
        # note:
        # senas can shrink the channels into 1/k
        # c_part cannot be too small in a challenging task or init_channels is already small
        self.k = 4
        self._meta_node_num = meta_node_num
        self._input_num = 2

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = build_rectify(c_in0, c_in1, cell_type)
            c_part = c_out // double_down
            c_part = int(c_part // self.k)
        else:
            self.preprocess0 = ShrinkBlock(c_in0, c_in1)
            c_part = c_out
            c_part = int(c_part // self.k)
        self.preprocess1 = build_activation(False)
        self.node_activation = build_activation()

        self.post_process = RectifyBlock(c_part * self._meta_node_num, c_out, cell_type=cell_type)

        self._ops = nn.ModuleList()
        # i=0  j=0,1
        # i=1  j=0,1,2
        # i=2  j=0,1,2,3
        # _ops=2+3+4=9
        for i in range(self._meta_node_num):
            for j in range(self._input_num + i):  # the input id for remaining meta-node
                # only the first input is reduction
                # down cell: |_|_|_|_|*|_|_|*|*| where _ indicate down operation
                # up cell:   |*|_|*|*|_|*|_|*|*| where _ indicate up operation
                if j < self._input_num:
                    if cell_type == 'down':
                        op = MixedOp(c_in1, c_part, OpType.DOWN)
                    elif j > 0:
                        op = MixedOp(c_in1, c_part, OpType.UP)
                    else:
                        op = MixedOp(c_in1, c_part, OpType.NORM)
                else:
                    op = MixedOp(c_part, c_part, OpType.NORM)
                self._ops += [op]

    def forward(self, in0, in1, weights_norm, weights_chg, betas):
        in0 = self.preprocess0(in0)
        in1 = self.preprocess1(in1)
        states = [in0, in1]
        offset = 0
        # offset=0  states=2  _ops=[0,1]
        # offset=2  states=3  _ops=[2,3,4]
        # offset=5  states=4  _ops=[5,6,7,8]
        for i in range(self._meta_node_num):
            # handle the un-consistent dimension
            edge_beta = betas[offset:(offset + len(states))]
            for j, h in enumerate(states):
                edge_feature = edge_beta[j] * self._ops[offset + j](h, weights_norm[offset + j], weights_chg[offset + j])
                node_feature = edge_feature if j == 0 else (node_feature + edge_feature)
            offset += len(states)
            s = self.node_activation(node_feature)
            states.append(s)

        return self.post_process(torch.cat(states[-self._meta_node_num:], dim=1))
