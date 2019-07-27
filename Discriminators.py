import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module


class Discr(Module):
	def __init__(self):
		super(Discr, self).__init__()
		self.fc1 = torch.nn.Linear(128, 128)
		self.fc2 = torch.nn.Linear(128,128)
		self.fc3 = torch.nn.Linear(128, 1)
		self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
		self.sig = nn.Sigmoid()
		
	def forward(self,X):
		X = torch.flatten(X, start_dim=1)
		out = F.relu(self.fc1(X))
		out = self.dropout(X)
		out = F.relu(self.fc2(X))
		out = self.fc3(X)
		out = self.sig(out)

		return out
		