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

class Model(Module):
	def __init__(self):
		super(Model, self).__init__()
		self.gen = HandWrtng()
		self.D1 = Discr()
		self.D2 = Discr()

		self.FTN = FTNet()

	def forward(self, X_t, X_s, X_sp):

		f_Xt = self.gen(X_t)
		f_Xs = self.gen(X_s)
		f_Xsp = self.gen(X_sp)
		f_Xt1 = f_Xt.detach()
		f_Xs1 = f_Xs.detach()
		f_Xsp1 = f_Xsp.detach()

		g_f_Xs = self.FTN(f_Xs1)
		g_f_Xsp = self.FTN(f_Xsp1)
		g_f_Xs1 = g_f_Xs.detach()
		g_f_Xsp1 = g_f_Xsp.detach()

		D2_f_Xt = self.D2(f_Xt1)
		D2_f_Xs = self.D2(f_Xs1)
		D2_g_f_Xs = self.D2(g_f_Xs1)

		D1_f_Xt = self.D1(f_Xt1)
		D1_g_f_Xs = self.D1(g_f_Xs1)

		return f_Xt, f_Xs, f_Xsp, g_f_Xs, g_f_Xsp, D2_f_Xt, D2_f_Xs, D2_g_f_Xs, D1_f_Xt, D1_g_f_Xs

