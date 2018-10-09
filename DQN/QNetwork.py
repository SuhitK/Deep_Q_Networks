from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


isCUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(model):
	if isinstance(model, nn.Linear):
		nn.init.xavier_normal_(model.weight.data)
		if model.bias is not None:
			nn.init.xavier_normal_(model.bias.data)


class CartPoleNetwork(nn.Module):
	def __init__(self):
		super(CartPoleNetwork, self).__init__()
		self.ffnn1 = nn.Linear(4, 64)
		self.ffnn2 = nn.Linear(64, 128)
		self.ffnn3 = nn.Linear(128, 2)

	def forward(self, x):
		x = F.relu(self.ffnn1(x))
		x = F.relu(self.ffnn2(x))
		x = F.relu(self.ffnn3(x))


class MountainCarNetwork(nn.Module):
	def __init__(self):
		super(MountainCarNetwork, self).__init__()
		self.ffnn1 = nn.Linear(2, 64)
		self.ffnn2 = nn.Linear(64, 128)
		self.ffnn3 = nn.Linear(128, 3)

	def forward(self, x):
		x = F.relu(self.ffnn1(x))
		x = F.relu(self.ffnn2(x))
		x = F.relu(self.ffnn3(x))


class QNetwork():
	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name):
		'''
		  Define your network architecture here. It is also a good idea to define any training operations
		  and optimizers here, initialize your variables, or alternately compile your model here.
		'''
		self.environment_name = environment_name

		# Define model according to environment_name
		if environment_name.equals('CartPole-v0'):
			self.policyModel = CartPoleNetwork()
			self.targetModel = CartPoleNetwork()
			self.lr = 1e-3
		else:
			self.policyModel = MountainCarNetwork()
			self.targetModel = MountainCarNetwork()
			self.lr = 1e-4

		# Set the GPU characteristics of the environment
		self.policyModel = self.policyModel.to(isCUDA)
		self.targetModel = self.targetModel.to(isCUDA)

		# Initialize the model weights
		self.policyModel.apply(weight_init)
		self.targetModel.load_state_dict(policyModel.state_dict())

		# Define the Loss function and the Optimizer for the model
		self.criterion = nn.MSELoss()
		self.optimizer = nn.optim.Adam(self.policyModel.parameters(), lr=self.lr, momentum=0.9)


	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		torch.save(self.policyModel.state_dict(), '{}_{}'.format(self.environment_name, suffix))

	def load_model(self, model_file):
		# Helper function to load an existing model.
		model_dict = torch.load(model_file)
		self.policyModel.load_state_dict(model_dict['state_dict'])
		self.optimizer.parameters = model_dict['optimizer_params']


	def load_model_weights(self, weight_file):
		# Helper funciton to load model weights.
		self.policyModel.load_state_dict(torch.load(weight_file))