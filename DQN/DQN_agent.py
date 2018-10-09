import gym
import pdb
import torch
import random
import numpy as np

from torch.autograd import Variable

from DQN.QNetwork import QNetwork
from DQN.replay_memory import Replay_Memory


class DQN_Agent():
	def __init__(self, args, memory_size=50000, burn_in=10000, render=False):
		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.args = args
		self.epsilon = 0.5
		self.greedy_epsilon = self.args.greedy_epsilon_init
		self.env = gym.make(self.args.env)
		self.init_state = self.env.reset()
		self.dqnNetwork = QNetwork(self.args.env)
		self.replay_memory = Replay_Memory(memory_size=memory_size, burn_in=burn_in)

		self.num_observations = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.n
		self.actions = range(self.num_actions)

		self.use_cuda = torch.cuda.is_available()

	def decay_greedy_epsilon(self):
		self.greedy_epsilon -= (self.args.greedy_epsilon_init - self.args.greedy_epsilon_stop) / self.args.greedy_epsilon_iter

	def epsilon_greedy_policy(self, state):
		# Creating epsilon greedy probabilities to sample from.
		# best_action = np.argmax(q_values)

		# policy = np.zeros((self.num_actions))
		# policy[:] = self.epsilon / self.num_actions
		# policy[best_action] = 1 - self.epsilon

		# return np.random.choice(actions, p=policy)

		possible_actions = []
		possible_actions.append(torch.LongTensor([[random.randrange(self.num_actions)]]))
		possible_actions.append(self.get_action(state))

		# return possible_actions[np.random.choice([0, 1], p=[self.greedy_epsilon, (1 - self.greedy_epsilon)])]
		return possible_actions[0]

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		best_action = np.argmax(q_values)

		policy = np.zeros((self.num_actions))
		policy[:] = self.greedy_epsilon / self.num_actions
		policy[best_action] = 1 - self.greedy_epsilon

		return np.random.choice(actions, p=policy)

	def get_action(self, state):
		state = Variable(state)
		output = self.dqnNetwork.forward(state)
		return output.detach().data.max(1)[1].cpu().view(1, 1)

	def get_state_tensor(self, state):
		return torch.from_numpy(state.reshape((-1, 4))).float()

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.

		# TODO: Model update code comes here and also predicting q for current state
		for episode in range(self.args.epi):
			state = self.get_state_tensor(self.env.reset())
			# steps = 0

			while True:
				if self.args.render:
					self.env.render()

				action = self.epsilon_greedy_policy(state)

				self.decay_greedy_epsilon()

				next_state, reward, is_terminal, info = self.env.step(action[0, 0])

				next_state = self.get_state_tensor(next_state)
				reward = torch.Tensor([reward])

				transition = (state, action, reward, next_state, torch.LongTensor([is_terminal]))
				self.replay_memory.append(transition)

				state = next_state
				# steps += 1

				self.train_QNetwork()

				if is_terminal:
					break

			if episode % 100 == 0:
				avg_reward = self.test()
				print('Episode: {}\tAvg Reward: {}'.format(episode, avg_reward))

	def train_QNetwork(self):
		self.dqnNetwork.optimizer.zero_grad()

		input_batch = self.replay_memory.sample_batch(batch_size=self.args.bsz)
		states, actions, rewards, next_states, terminals = zip(*input_batch)

		states, actions, rewards, terminals = map(lambda x: Variable(torch.cat(x)), [states, actions, rewards, terminals])

		pred_values = torch.gather(self.dqnNetwork.forward(states), 1, actions)
		next_values = Variable(torch.zeros(self.args.bsz))

		non_terminal_next_states = Variable(torch.cat([state for idx, state in enumerate(next_states) if terminals[idx].data.cpu().numpy()[0] == 0]))
		next_values[terminals == 0] = self.dqnNetwork.forward(non_terminal_next_states).max(1)[0].detach()
		true_values = rewards + (next_values * self.args.gamma)

		loss = self.dqnNetwork.criterion(pred_values, true_values)

		if loss is not None:
			loss.backward()
			self.dqnNetwork.optimizer.step()

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		reward_sum = 0
		for i in range(self.args.test_epi):
			state = self.get_state_tensor(self.env.reset())

			while True:
				if self.args.render:
					self.env.render()

				action = self.get_action(state)

				next_state, reward, is_terminal, info = self.env.step(action[0, 0])
				state = self.get_state_tensor(next_state)
				reward_sum += reward

				if is_terminal:
					break

		return reward_sum / self.args.test_epi

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		state = self.get_state_tensor(self.env.reset())

		for _ in range(self.replay_memory.burn_in):
			action = torch.LongTensor([[random.randrange(self.num_actions)]])
			next_state, reward, is_terminal, info = self.env.step(action[0, 0])

			next_state = self.get_state_tensor(next_state)
			reward = torch.Tensor([reward])

			transition = (state, action, reward, next_state, torch.LongTensor([is_terminal]))

			self.replay_memory.append(transition)
			state = next_state

			if is_terminal:
				state = self.get_state_tensor(self.env.reset())
