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


useCUDA = torch.cuda.is_available()

def weight_init(model):
	if isinstance(model, nn.Linear):
		nn.init.xavier_normal_(model.weight.data)
		# if model.bias is not None:
		#       nn.init.xavier_normal_(model.bias.data)


class CartPoleNetwork(nn.Module):
	def __init__(self, duel):
		super(CartPoleNetwork, self).__init__()
		self.duel = duel
		self.ffnn1 = nn.Linear(4, 128)
		self.ffnn2 = nn.Linear(128, 128)
		self.ffnn3 = nn.Linear(128, 128)
		self.ffnn4 = nn.Linear(128, 2)

		if self.duel:
			self.value = nn.Linear(128, 1)

	def forward(self, x):
		x = F.relu(self.ffnn1(x))
		x = F.relu(self.ffnn2(x))
		x = F.relu(self.ffnn3(x))
		q_value = self.ffnn4(x)

		if self.duel:
			value = self.value(x)
			q_value = value.expand_as(q_value) + (q_value - q_value.mean(1, keepdim=True).expand_as(q_value))

		return q_value


class MountainCarNetwork(nn.Module):
	def __init__(self, duel):
		super(MountainCarNetwork, self).__init__()
		self.duel = duel
		self.ffnn1 = nn.Linear(2, 64)
		self.ffnn2 = nn.Linear(64, 128)
		self.ffnn3 = nn.Linear(128, 512)
		self.ffnn4 = nn.Linear(512, 128)
		self.ffnn5 = nn.Linear(128, 3)

		if self.duel:
			self.value = nn.Linear(128, 1)

	def forward(self, x):
		x = F.relu(self.ffnn1(x))
		x = F.relu(self.ffnn2(x))
		x = F.relu(self.ffnn3(x))
		x = F.relu(self.ffnn4(x))

		q_value = self.ffnn5(x)

		if self.duel:
			value = self.value(x)
			q_value = value.expand_as(q_value) + (q_value - q_value.mean(1, keepdim=True).expand_as(q_value))

		return q_value


class QNetwork():
	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name, duel=False):
		'''
		  Define your network architecture here. It is also a good idea to define any training operations
		  and optimizers here, initialize your variables, or alternately compile your model here.
		'''
		self.environment_name = environment_name

		# Define model according to environment_name
		if environment_name == 'CartPole-v0':
			self.policyModel = CartPoleNetwork(duel)
			self.targetModel = CartPoleNetwork(duel)
			self.lr = 1e-4
		else:
			self.policyModel = MountainCarNetwork(duel)
			self.targetModel = MountainCarNetwork(duel)
			self.lr = 5e-4

		# Set the GPU characteristics of the environment
		if useCUDA:
			self.policyModel = self.policyModel.cuda()
			self.targetModel = self.targetModel.cuda()

		# Initialize the model weights
		self.policyModel.apply(weight_init)
		self.equate_target_model_weights()

		# Define the Loss function and the Optimizer for the model
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.policyModel.parameters(), lr=self.lr, weight_decay=1e-2)
		# self.optimizer = torch.optim.RMSprop(self.policyModel.parameters(), lr=self.lr, weight_decay=1e-2, momentum=0.9)

	def forward(self, input_vector, mode='policyModel'):
		return self.policyModel(input_vector) if mode == 'policyModel' else self.targetModel(input_vector)

	def equate_target_model_weights(self):
		self.targetModel.load_state_dict(self.policyModel.state_dict())

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		torch.save(self.policyModel.state_dict(), 'PolicyModel/{}'.format(suffix))
		torch.save(self.policyModel.state_dict(), 'TargetModel/{}'.format(suffix))

	def load_model(self, model_file):
		# Helper function to load an existing model.
		model_dict = torch.load(model_file)
		self.policyModel.load_state_dict(model_dict['state_dict'])
		self.optimizer.parameters = model_dict['optimizer_params']

	def load_model_weights(self, weight_file):
		# Helper funciton to load model weights.
		if weight_file is not None:
			self.policyModel.load_state_dict(torch.load(weight_file))
			self.equate_target_model_weights()

	def print_model(self):
		print(self.policyModel)
		print('\n\n')

	def print_model_summary(self, *input_size):
		try:
			from torchsummary import summary
			summary(self.policyModel, *input_size)
			print('\n\n')
		except ImportError as e:
			pass
