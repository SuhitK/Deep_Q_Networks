import csv
import gym
import pdb
import math
import torch
import random
import numpy as np

from datetime import datetime
from torch.autograd import Variable

from DQN.QNetwork import QNetwork
from DQN.replay_memory import Replay_Memory

OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
ENDC = '\033[0m'


class DQN_Agent():
	def __init__(self, args, memory_size=50000, burn_in=10000, render=False):
		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.args = args
		self.epsilon = args.epsilon_init
		self.greedy_epsilon = self.args.greedy_epsilon
		self.env = gym.make(self.args.env)
		self.env = gym.wrappers.Monitor(self.env, self.args.folder_prefix + self.args.env, force=True)
		self.dqnNetwork = QNetwork(self.args.env, self.args.duel_dqn)
		self.replay_memory = Replay_Memory(memory_size=memory_size, burn_in=burn_in)
		self.dqnNetwork.load_model_weights(self.args.weight_file)

		self.num_observations = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.n
		self.actions = range(self.num_actions)
		self.decay = (self.epsilon - self.args.epsilon_stop) / self.args.epsilon_iter

		self.use_cuda = torch.cuda.is_available()
		self.env_is_terminal = True
		self.state = None
		# self.env.seed(0)

		self.dqnNetwork.print_model()
		self.dqnNetwork.print_model_summary((self.args.bsz, self.num_observations))

	def load_model(self, model_file, weights=False):
		if weights:
			self.dqnNetwork.load_model_weights(model_file)
		else:
			self.dqnNetwork.load_model(model_file)

	def decay_epsilon(self):
		if self.args.decay:
			self.epsilon *= self.decay
		self.epsilon = max(self.epsilon, self.args.epsilon_stop)

	def epsilon_greedy_policy(self, state):
		possible_actions = []
		possible_actions.append(torch.LongTensor([[random.randrange(self.num_actions)]]))
		possible_actions.append(self.get_action(state))

		return possible_actions[np.random.choice([0, 1], p=[self.epsilon, (1 - self.epsilon)])]

	def greedy_policy(self, state):
		possible_actions = []
		possible_actions.append(torch.LongTensor([[random.randrange(self.num_actions)]]))
		possible_actions.append(self.get_action(state))

		return possible_actions[np.random.choice([0, 1], p=[self.greedy_epsilon, (1 - self.greedy_epsilon)])]

	def get_action(self, state, mode='policyModel'):
		state = Variable(state)
		output = self.dqnNetwork.forward(state, mode=mode)
		return output.detach().data.max(1)[1].cpu().view(1, 1)

	def get_cartpole_lookahead_step(self, action, state):
		# CartPole Step definition (copied as is from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
		x, x_dot, theta, theta_dot = state
		force = self.env.env.env.force_mag if action==1 else -self.env.env.env.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)
		temp = (force + self.env.env.env.polemass_length * theta_dot * theta_dot * sintheta) / self.env.env.env.total_mass
		thetaacc = (self.env.env.env.gravity * sintheta - costheta* temp) / (self.env.env.env.length * \
		                                                                 (4.0/3.0 - self.env.env.env.masspole * costheta * costheta / self.env.env.env.total_mass))
		xacc  = temp - self.env.env.env.polemass_length * thetaacc * costheta / self.env.env.env.total_mass
		if self.env.env.env.kinematics_integrator == 'euler':
			x  = x + self.env.env.env.tau * x_dot
			x_dot = x_dot + self.env.env.env.tau * xacc
			theta = theta + self.env.env.env.tau * theta_dot
			theta_dot = theta_dot + self.env.env.env.tau * thetaacc
		else: # semi-implicit euler
			x_dot = x_dot + self.env.env.env.tau * xacc
			x  = x + self.env.env.env.tau * x_dot
			theta_dot = theta_dot + self.env.env.env.tau * thetaacc
			theta = theta + self.env.env.env.tau * theta_dot
		state = (x,x_dot,theta,theta_dot)
		done =  x < -self.env.env.env.x_threshold \
				or x > self.env.env.env.x_threshold \
				or theta < -self.env.env.env.theta_threshold_radians \
				or theta > self.env.env.env.theta_threshold_radians
		done = bool(done)

		if not done:
			reward = 1.0
		elif self.env.env.env.steps_beyond_done is None:
			# Pole just fell!
			reward = 1.0
		else:
			if self.env.env.env.steps_beyond_done == 0:
				print(WARNING + "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior." + ENDC)
			reward = 0.0

		return np.array(state), reward, done, {}

	def get_cartpole_lookahead_action(self, action):
		state_1 = self.state.cpu().numpy()[0]
		max_Qval_1, max_action_1 = float('-inf'), float('-inf')
		for action_1 in range(self.num_actions):
			state_2, reward_1, done_1, _ = self.get_cartpole_lookahead_step(action_1, state_1)

			if done_1:
				if max_Qval_1 < reward_1:
					max_Qval_1, max_action_1 = reward_1, action_1
				continue

			max_Qval_2 = float('-inf')
			for action_2 in range(self.num_actions):
				state_3, reward_2, done_2, _ = self.get_cartpole_lookahead_step(action_2, state_2)

				if done_2:
					if max_Qval_2 < reward_2:
						max_Qval_2= reward_2
					continue

				Qval_2 = (reward_2 + self.args.gamma * self.dqnNetwork.forward(self.map_cuda(self.get_state_tensor(state_3)))).max(1)[0]
				if max_Qval_2 < Qval_2:
					max_Qval_2 = Qval_2

			Qval_1 = (reward_1 + self.args.gamma * max_Qval_2)
			if max_Qval_1 < Qval_1:
				max_Qval_1 = Qval_1
				max_action_1 = action_1

		return torch.LongTensor([[max_action_1]])

	def get_init_state(self):
		if self.env_is_terminal:
			state = self.map_cuda(self.get_state_tensor(self.env.reset()))
		else:
			action = self.env.action_space.sample()
			state, reward, self.env_is_terminal, info = self.env.step(action)
			state = self.map_cuda(self.get_state_tensor(state))
			if self.env_is_terminal:
				# state = self.map_cuda(self.get_state_tensor(self.env.reset()))
				state = self.get_init_state()
		return state

	def get_state_tensor(self, state):
		return torch.from_numpy(state.reshape((-1, self.num_observations))).float()

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
		avgRewardFilename = "{}RewardsCSV/Average_Rewards_{}_{}.csv".format(self.args.folder_prefix, self.args.env, time)
		avgRewardFile = open(avgRewardFilename, 'w')
		avg_reward = 0
		steps = 0

		if self.args.lookahead and self.args.env == 'CartPole-v0':
			lookaheadFilename = "{}RewardsCSV/Average_Rewards_2Step_Lookahead_{}_{}.csv".format(self.args.folder_prefix, self.args.env, time)
			lookaheadFile = open(lookaheadFilename, 'w')
			lookahead_reward = 0

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

			if episode % self.args.target_update == self.args.target_update -1:
				self.dqnNetwork.equate_target_model_weights()

			if episode % self.args.test_every == self.args.test_every - 1:
				avg_reward = self.test(lookahead=self.greedy_policy)
				avgRewardFile.write('{}\n'.format(avg_reward))

				if self.args.lookahead and self.args.env == 'CartPole-v0':
					print(OKBLUE + 'Two Step Look Ahead Test' + ENDC)
					lookahead_reward = self.test(lookahead=self.get_cartpole_lookahead_action)
					lookaheadFile.write('{}\n'.format(lookahead_reward))
					print(OKGREEN + 'Train Episode: {}\tAvg. Test Reward: {}\t Avg. 2 Step Lookahead Reward: {}'.format(episode+1, avg_reward, lookahead_reward) + ENDC)
				else:
					print(OKGREEN + 'Train Episode: {}\tAvg. Test Reward: {}'.format(episode+1, avg_reward) + ENDC)

			if (episode % self.args.save_epi == self.args.save_epi - 1) or (avg_reward > 190.0 and episode % 100 == 99):
				self.dqnNetwork.save_model_weights(prefix=self.args.folder_prefix, suffix='{}_epi{}_rew{:.4f}_{}.pkl'.format(self.args.env, episode+1, avg_reward, time))

		avg_reward = self.test(test_epi=100, lookahead=self.greedy_policy)

		if self.args.lookahead and self.args.env == 'CartPole-v0':
			lookahead_reward = self.test(test_epi=100, lookahead=self.get_cartpole_lookahead_action)
			print(OKGREEN + 'Trained Model - Avg. Test Reward: {}\t Avg. 2 Step Lookahead Reward: {}'.format(avg_reward, lookahead_reward) + ENDC)
		else:
			print(OKGREEN + 'Trained Model - Avg. Test Reward: {}'.format(avg_reward) + ENDC)

		avgRewardFile.close()

	def train_QNetwork(self):
		self.dqnNetwork.optimizer.zero_grad()

		input_batch = self.replay_memory.sample_batch(batch_size=self.args.bsz)
		states, actions, rewards, next_states, terminals = zip(*input_batch)

		states, actions, rewards, terminals = map(lambda x: self.map_cuda(Variable(torch.cat(x))), [states, actions, rewards, terminals])

		pred_action = self.dqnNetwork.forward(states)
		pred_values = torch.gather(pred_action, 1, actions)
		next_values = self.map_cuda(Variable(torch.zeros(self.args.bsz)))

		if self.args.double_dqn:
			actions = pred_action.data.max(1)[1].view(-1, 1)
			next_values = self.dqnNetwork.forward(self.map_cuda(Variable(torch.cat(next_states))), mode='targetModel')
			next_values = torch.gather(next_values, 1, Variable(actions)).detach().view(-1)
			next_values[terminals == 1] = 0
		else:
			non_terminal_next_states = Variable(torch.cat([state for idx, state in enumerate(next_states) if terminals.data.cpu().numpy()[idx] == 0]))
			next_values[terminals == 0] = self.dqnNetwork.forward(non_terminal_next_states, mode='targetModel').max(1)[0].detach()
		true_values = (rewards + (next_values * self.args.gamma)).view(-1, 1)

		loss = self.dqnNetwork.criterion(pred_values, true_values)

		if loss is not None:
			loss.backward()
			self.dqnNetwork.optimizer.step()

	def test(self, test_epi=None, model_file=None, lookahead=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		if test_epi is None:
			test_epi = self.args.test_epi

		reward_sum = 0
		for i in range(test_epi):
			state = self.get_init_state()
			episode_steps = 0

			while True:
				if self.args.render:
					self.env.render()

				self.state = state
				action = self.map_cuda(lookahead(state))

				next_state, reward, self.env_is_terminal, info = self.env.step(action.cpu().numpy()[0, 0])
				state = self.map_cuda(self.get_state_tensor(next_state))
				reward_sum += reward
				episode_steps += 1

				if self.env_is_terminal or (self.args.env == 'CartPole-v0' and episode_steps == 200):
					break
			print('Steps: {}'.format(episode_steps))

		return np.mean(self.env.get_episode_rewards()[-test_epi:])
		# return reward_sum / test_epi

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