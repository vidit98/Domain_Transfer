import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module

class HandWrtng(Module):
	def __init__(self):
		super(HandWrtng, self).__init__()
		self.conv1_1 = torch.nn.Conv2d(3, 32, 3, padding=1)
		self.conv1_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
		
		self.maxpool = torch.nn.MaxPool2d(2)
		
		self.conv2_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
		self.conv2_2 = torch.nn.Conv2d(64, 64, 3, padding=1)

		self.conv3_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
		self.conv3_2 = torch.nn.Conv2d(128, 128, 3, padding=1)

		self.fc1 = torch.nn.Linear(4*4*128, 128)
		self.fc2 = torch.nn.Linear(128,128)

	def forward(self, X):
		out = F.relu(self.conv1_1(X))
		out = F.relu(self.conv1_2(out))
		out = self.maxpool(out)

		out = F.relu(self.conv2_1(out))
		out = F.relu(self.conv2_2(out))
		out = self.maxpool(out)

		out = F.relu(self.conv3_1(out))
		out	= F.relu(self.conv3_2(out))
		out = self.maxpool(out)
	
		out = torch.flatten(out, start_dim=1)
		out = F.relu(self.fc1(out))
		out = self.fc2(out)

		norm = 0.5*out**2
		norm = torch.sqrt(torch.sum(norm))

		out = out/norm

		return out


