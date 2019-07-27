import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module


class FTNet(Module):
	def __init__(self):
		super(FTNet, self).__init__()
		self.fc1 = torch.nn.Linear(128, 256)
		self.fc2 = torch.nn.Linear(256,256)
		self.fc3 = torch.nn.Linear(256, 128)
		self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
		
	def forward(self,X):
		identity = X
		X = torch.flatten(X, start_dim=1)
		out = F.relu(self.fc1(X))
		out = self.dropout(out)
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		out += identity

		return out


		
		