import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module

from Generator import HandWrtng
from Discriminators import Discr
from FTN import FTNet

lambda1 = 0.3
lambda2 = 0.03

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg



class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss

def cross_entropy(logits, target, size_average=True):
    
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))




def loss(f_Xt, f_Xs, f_Xsp, g_f_Xs, g_f_Xsp, target, D2_f_Xt, D2_f_Xs, D2_g_f_Xs, D1_f_Xt, D1_g_f_Xs):

    npair = NpairLoss(l2_reg=0).to(device)
    lf = 0.5*npair(f_Xs, f_Xsp, target)
    lf += 0.5*npair(g_f_Xs.detach(), g_f_Xsp.detach(), target)
    lf += lambda1*torch.mean(torch.log(D1_f_Xt.detach()))
    lf += lambda2*(torch.mean(torch.log(D2_f_Xs.detach())) + 0.5*(torch.mean(torch.log(1-D2_g_f_Xs.detach())) + torch.mean(torch.log(1-D2_f_Xt.detach()))))

    lg = 0.5*npair(g_f_Xs, g_f_Xsp, target)
    lg += lambda2*torch.mean(torch.log(1 - D2_g_f_Xs.detach()))

    lD1 = torch.mean(torch.log(D1_g_f_Xs)) + torch.mean(torch.log(1 - D1_f_Xt))
    lD2 = lambda2*(torch.mean(torch.log(D2_f_Xs)) + 0.5*(torch.mean(torch.log(1-D2_g_f_Xs)) + torch.mean(torch.log(1-D2_f_Xt))))
    #print(D2_f_Xt, D2_f_Xs, D2_g_f_Xs, D1_f_Xt, D1_g_f_Xs)
    # print(' D2fxt: {:.6f}, D2fxs: {:.6f},D2gfxs: {:.6f},D1fxt: {:.6f}, D1gfXs: {:.6f}'
    #               .format(D2_f_Xt[0][0], D2_f_Xs[0][0], D2_g_f_Xs[0][0], D1_f_Xt[0][0], D1_g_f_Xs[0][0]))
    return lf, lg, lD1, lD2