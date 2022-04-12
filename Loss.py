import torch
import torch.nn as nn
from torch.nn import functional as F


class Loss(object):

    def __init__(self, args):
        self.args = args
        self.mu = args.mu
        self.xi = args.xi
        self.alpha = args.alpha
        self.beta = args.beta
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.args.device)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)

    def _loss_dist(self, net1_param, net2_param):
        loss = 0
        for param1, param2 in zip(net1_param, net2_param):
            loss += torch.nn.functional.mse_loss(param1, param2)
        return loss**self.mu

    def _loss_classification(self, X, Y, reduction='mean'):
        if reduction == 'mean':
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))
        elif reduction == 'none':
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))), dim=1)
        elif reduction == 'sum':
            return torch.sum(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))
        else:
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))

    def _loss_entropy(self, X, reduction='mean'):
        if reduction == 'mean':
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)))
        elif reduction == 'none':
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)), dim=1)
        elif reduction == 'sum':
            return - torch.sum(torch.mul(self.softmax(X), self.logsoftmax(X)))
        else:
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)))

    def _loss_compatibility(self, Y, T, reduction='mean'):
        return F.cross_entropy(Y, T, reduction=reduction)

    def _pencil_loss(self, X, last_y_var_A, alpha, beta, reduction='mean', target_var=None):
        assert not target_var == None
        # lc is classification loss
        lc = self._loss_classification(X, last_y_var_A, reduction=reduction)
        # le is entropy loss
        le = self._loss_entropy(X, reduction=reduction)
        # lo is compatibility loss
        lo = self._loss_compatibility(last_y_var_A, target_var, reduction=reduction)
        return lc + alpha * lo + beta * le

    def get_loss(self, X, Y, loss_type='CE', reduction='mean', **kwargs):
        if loss_type == 'CE':
            loss = F.cross_entropy(X, Y, reduction=reduction)
        elif loss_type == 'KL':
            loss = F.kl_div(X, Y, reduction=reduction)
        elif loss_type == "FLIP_KL":
            loss = self._loss_classification(X, Y, reduction=reduction)
        elif loss_type == 'PENCIL':
            loss = self._pencil_loss(X, Y, alpha=self.alpha, beta=self.beta, reduction=reduction, **kwargs)
        else:
            loss = F.cross_entropy(X, Y, reduction=reduction)
        return loss

    def _sort_by_loss(self, predict, target, loss_type='CE', index=True, **kwargs):
        loss = self.get_loss(predict, target, loss_type=loss_type, reduction='none', **kwargs)
        index_sorted = torch.argsort(loss.data.cpu()).numpy()
        return index_sorted if index else predict[index_sorted]

    # Loss functions
    def loss_mlc(self, y_1, y_2, t_1, t_2, forget_rate, loss_type='CE',
                 target_var=None, softmax=False, net_param1=None, net_param2=None):
        if softmax:
            y_1 = self.softmax(y_1)
            y_2 = self.softmax(y_2)

        # compute NetA prediction loss
        ind_1_sorted = self._sort_by_loss(y_1, t_1, loss_type=loss_type, index=True, target_var=target_var)

        # compute NetB prediction loss
        ind_2_sorted = self._sort_by_loss(y_2, t_2, loss_type=loss_type, index=True, target_var=target_var)

        # catch R(t)% samples
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        # exchange
        loss_1_update = self.get_loss(
            y_1[ind_2_update],
            t_2[ind_2_update],
            loss_type=loss_type,
            reduction='none',
            target_var=None if target_var == None else target_var[ind_2_update]
        )
        loss_2_update = self.get_loss(
            y_2[ind_1_update],
            t_1[ind_1_update],
            loss_type=loss_type,
            reduction='none',
            target_var=None if target_var == None else target_var[ind_1_update]
        )

        ld = self._loss_dist(net_param1, net_param2)
        loss_1_final = torch.sum(loss_1_update) / num_remember + self.xi * ld
        loss_2_final = torch.sum(loss_2_update) / num_remember + self.xi * ld

        return loss_1_final, loss_2_final