import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz, p_task, p_meta, num_class=5):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config

        # specify the percentage of selected filters here:
        assert p_task != None and p_meta != None
        self.p_task = p_task
        self.p_meta = p_meta

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.masks_trn = []
        self.masks_val = []
        self.param_conv = nn.ParameterList()  # used for the meta-learner only
        self.param_other = nn.ParameterList()  # used for the meta-learner only
        self.num_class = num_class
        self.is_conv = []

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)  # weights
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  # bias

                # masks
                self.masks_trn.append(np.zeros(param[0], dtype=np.float32))
                self.masks_val.append(np.zeros(param[0], dtype=np.float32))
                self.param_conv.append(w)
                self.param_conv.append(self.vars[-1])
                self.is_conv.extend([1, 1])

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.masks_trn.append(np.zeros(param[1], dtype=np.float32))
                self.masks_val.append(np.zeros(param[1], dtype=np.float32))
                self.param_conv.append(w)
                self.param_conv.append(self.vars[-1])
                self.is_conv.extend([1, 1])

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.param_other.append(w)
                self.param_other.append(self.vars[-1])
                self.is_conv.extend([0, 0])

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                self.param_other.append(w)
                self.param_other.append(self.vars[-1])
                self.is_conv.extend([0, 0])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

        self.num_samples_tracked = nn.Parameter(torch.zeros(1), requires_grad=False)

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        self.num_samples_tracked += x.shape[0]
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training,
                                 momentum=x.shape[0] / self.num_samples_tracked.item())
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def compute_mask(self, selection='train', vars=None):
        """
        Compute mask for the input weights/parameters
        by selecting certain filters/feature maps
        using BN scaling factor
        """
        idx = 0
        mask_idx = 0

        if vars is None:
            vars = self.vars

        if selection == 'train':
            masks = self.masks_trn
            p = self.p_task
        elif selection == 'val':
            masks = self.masks_val
            p = self.p_meta

        with torch.no_grad():
            bn_count = 0
            for name, param in self.config:
                if name == 'bn':
                    num_filter = param[0]
                    gamma = vars[idx]
                    selected = torch.topk(gamma, int(num_filter * p[bn_count]))[1].cpu().numpy()
                    masks[mask_idx][selected] = 1

                    idx += 2
                    mask_idx += 1
                    bn_count += 1

                elif name == 'linear' or name == 'conv2d' or name == 'convt2d':
                    idx += 2

    def reset_mask(self, selection='train'):
        """
        Reset the mask to be all 0.
        """
        if selection == 'train':
            for m in self.masks_trn:
                m.fill(0)
        elif selection == 'val':
            for m in self.masks_val:
                m.fill(0)
        elif selection == 'both':
            for m1, m2 in zip(self.masks_trn, self.masks_val):
                m1.fill(0)
                m2.fill(0)

    def apply_mask(self, selection='train', grad=None):
        """
        Apply the mask to the weights' gradient
        """
        idx = 0
        mask_idx = 0
        if selection == 'train':
            masks = self.masks_trn
        elif selection == 'val':
            masks = self.masks_val

        # update the meta-learner?
        if grad is None:
            vars = self.vars
            with torch.no_grad():
                for name, _ in self.config:
                    if name == 'conv2d':
                        w_grad, b_grad = vars[idx].grad, vars[idx + 1].grad
                        w_grad.mul_(torch.Tensor(masks[mask_idx][:, None, None, None]).cuda())
                        b_grad.mul_(torch.Tensor(masks[mask_idx]).cuda())
                        idx += 2
                        mask_idx += 1
                    elif name == 'convt2d':
                        w_grad, b_grad = vars[idx].grad, vars[idx + 1].grad
                        w_grad.mul_(torch.Tensor(masks[mask_idx][None, :, None, None]).cuda())
                        b_grad.mul_(torch.Tensor(masks[mask_idx]).cuda())
                        idx += 2
                        mask_idx += 1
                    elif name == 'linear' or name == 'bn':
                        idx += 2

        # update the individual task learner
        else:
            with torch.no_grad():
                for name, _ in self.config:
                    if name == 'conv2d':
                        w_grad, b_grad = grad[idx], grad[idx + 1]
                        w_grad.mul_(torch.Tensor(masks[mask_idx][:, None, None, None]).cuda())
                        b_grad.mul_(torch.Tensor(masks[mask_idx]).cuda())
                        idx += 2
                        mask_idx += 1
                    elif name == 'convt2d':
                        w_grad, b_grad = grad[idx], grad[idx + 1]
                        w_grad.mul_(torch.Tensor(masks[mask_idx][None, :, None, None]).cuda())
                        b_grad.mul_(torch.Tensor(masks[mask_idx]).cuda())
                        idx += 2
                        mask_idx += 1
                    elif name == 'linear' or name == 'bn':
                        idx += 2

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
