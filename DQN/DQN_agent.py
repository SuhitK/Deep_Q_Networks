import gym
import pdb
import torch
import random
import numpy as np
import csv
from datetime import datetime

from torch.autograd import Variable

from DQN.QNetwork import QNetwork
from DQN.replay_memory import Replay_Memory


class DQN_Agent():
	def __init__(self, args, memory_size=50000, burn_in=10000, render=False):
		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.args = args
		self.epsilon = args.epsilon_init
		self.greedy_epsilon = self.args.greedy_epsilon
		self.env = gym.make(self.args.env)
		self.env = gym.wrappers.Monitor(self.env, self.args.env, force=True)
		self.dqnNetwork = QNetwork(self.args.env)
		self.replay_memory = Replay_Memory(memory_size=memory_size, burn_in=burn_in)

		self.num_observations = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.n
		self.actions = range(self.num_actions)
		self.decay = (self.epsilon - self.args.epsilon_stop) / self.args.epsilon_iter

		self.use_cuda = torch.cuda.is_available()
		self.env_is_terminal = True

	def decay_epsilon(self):
		self.epsilon *= self.decay
		self.epsilon = max(self.epsilon, self.args.epsilon_stop)

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

		return possible_actions[np.random.choice([0, 1], p=[self.epsilon, (1 - self.epsilon)])]

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		# best_action = np.argmax(q_values)

		# policy = np.zeros((self.num_actions))
		# policy[:] = self.greedy_epsilon / self.num_actions
		# policy[best_action] = 1 - self.greedy_epsilon

		# return np.random.choice(actions, p=policy)
		possible_actions = []
		possible_actions.append(torch.LongTensor([[random.randrange(self.num_actions)]]))
		possible_actions.append(self.get_action(state))

		return possible_actions[np.random.choice([0, 1], p=[self.greedy_epsilon, (1 - self.greedy_epsilon)])]


	def get_action(self, state, mode='policyModel'):
		state = Variable(state)
		output = self.dqnNetwork.forward(state, mode=mode)
		return output.detach().data.max(1)[1].cpu().view(1, 1)

	def get_init_state(self):
		if self.env_is_terminal:
			state = self.map_cuda(self.get_state_tensor(self.env.reset()))
		else:
			action = self.env.action_space.sample()
			state, reward, is_terminal, info = self.env.step(action)
			state = self.map_cuda(self.get_state_tensor(state))
			if is_terminal:
				state = self.map_cuda(self.get_state_tensor(self.env.reset()))
		return state

	def get_state_tensor(self, state):
		return torch.from_numpy(state.reshape((-1, 4))).float()

	def map_cuda(self, tensor):
		return tensor.cuda() if self.use_cuda else tensor

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.

		# TODO: Model update code comes here and also predicting q for current state
		time = datetime.now().time()
		avgRewardFilename = "RewardsCSV/Average_Rewards_{}_{}.csv".format(self.args.env, time)
		avgRewardFile = open(avgRewardFilename, 'w')
		steps = 0

		for episode in range(self.args.epi):
			state = self.get_init_state()
			episode_steps = 0

			while True:
				if self.args.render:
					self.env.render()

				action = self.map_cuda(self.epsilon_greedy_policy(state))
				self.decay_epsilon()

				next_state, reward, self.env_is_terminal, info = self.env.step(action.cpu().numpy()[0, 0])

				next_state = self.map_cuda(self.get_state_tensor(next_state))
				terminal = self.map_cuda(torch.LongTensor([self.env_is_terminal]))
				reward = self.map_cuda(torch.Tensor([reward]))

				transition = (state, action, reward, next_state, terminal)
				self.replay_memory.append(transition)

				state = next_state
				episode_steps += 1
				steps += 1

				self.train_QNetwork()

				if self.env_is_terminal or (self.args.env == 'CartPole-v0' and episode_steps == 200):
					break

			# if (episode < 500 and episode % 50 == 49) or episode % 500 == 499:
			# 		self.dqnNetwork.equate_target_model_weights()

			if episode % self.args.test_epi == self.args.test_epi - 1:
				self.dqnNetwork.equate_target_model_weights()
				avg_reward = self.test()
				avgRewardFile.write('{}\n'.format(avg_reward))
				print('Episode: {}\tAvg Reward: {}'.format(episode+1, avg_reward))

			if (episode % self.args.save_epi == self.args.save_epi - 1) or (avg_reward > 190.0):
				self.dqnNetwork.save_model_weights(suffix='{}_epi{}_rew{:.4f}_{}.pkl'.format(self.args.env, episode+1, avg_reward, time))

		avgRewardFile.close()

	def train_QNetwork(self):
		self.dqnNetwork.optimizer.zero_grad()

		input_batch = self.replay_memory.sample_batch(batch_size=self.args.bsz)
		states, actions, rewards, next_states, terminals = zip(*input_batch)

		states, actions, rewards, terminals = map(lambda x: self.map_cuda(Variable(torch.cat(x))), [states, actions, rewards, terminals])

		pred_values = torch.gather(self.dqnNetwork.forward(states), 1, actions)
		next_values = self.map_cuda(Variable(torch.zeros(self.args.bsz)))

		non_terminal_next_states = Variable(torch.cat([state for idx, state in enumerate(next_states) if terminals.data.cpu().numpy()[idx] == 0]))
		next_values[terminals == 0] = self.dqnNetwork.forward(non_terminal_next_states, mode = 'targetModel').max(1)[0].detach()
		true_values = (rewards + (next_values * self.args.gamma)).view(-1, 1)

		loss = self.dqnNetwork.criterion(pred_values, true_values)

		if loss is not None:
			loss.backward()
			self.dqnNetwork.optimizer.step()

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		reward_sum = 0
		for i in range(self.args.test_epi):
			state = self.get_init_state()
			episode_steps = 0

			while True:
				if self.args.render:
					self.env.render()

				action = self.map_cuda(self.get_action(state))

				next_state, reward, self.env_is_terminal, info = self.env.step(action.cpu().numpy()[0, 0])
				state = self.map_cuda(self.get_state_tensor(next_state))
				reward_sum += reward
				episode_steps += 1

				if self.env_is_terminal or (self.args.env == 'CartPole-v0' and episode_steps == 200):
					break
			print('Steps: {}'.format(episode_steps))

		return np.mean(self.env.get_episode_rewards()[-100:])
		# return reward_sum / self.args.test_epi

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		state = self.get_init_state()

		for _ in range(self.replay_memory.burn_in):
			action = self.map_cuda(torch.LongTensor([[random.randrange(self.num_actions)]]))
			next_state, reward, self.env_is_terminal, info = self.env.step(action.cpu().numpy()[0, 0])

			next_state = self.map_cuda(self.get_state_tensor(next_state))
			terminal = self.map_cuda(torch.LongTensor([self.env_is_terminal]))
			reward = self.map_cuda(torch.Tensor([reward]))

			transition = (state, action, reward, next_state, terminal)
			self.replay_memory.append(transition)

			state = next_state

			if self.env_is_terminal:
				state = self.map_cuda(self.get_state_tensor(self.env.reset()))

	def agent_close(self):
		self.env.env.close()