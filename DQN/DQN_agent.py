import gym
import numpy as np
import pdb

from DQN.replay_memory import Replay_Memory
from DQN.QNetwork import QNetwork


class DQN_Agent():
	def __init__(self, args, memory_size=50000, burn_in=10000, render=False):
		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.args = args
		self.epsilon = 0.5              # Epsilon used for epsilon-greedy policy
		self.greedy_epsilon = 0.05      # Epsilon used for greedy policy
		self.env = gym.make(self.args.env)
		self.init_state = self.env.reset()
		self.dqnNetwork = QNetwork(self.args.env)
		self.replay_memory = Replay_Memory(memory_size=memory_size, burn_in=burn_in)

		self.num_observations = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.n
		self.actions = range(self.num_actions)

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		best_action = np.argmax(q_values)

		policy = np.zeros((self.num_actions))
		policy[:] = self.epsilon / self.num_actions
		policy[best_action] = 1 - self.epsilon

		return np.random.choice(actions, p=policy)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		best_action = np.argmax(q_values)

		policy = np.zeros((self.num_actions))
		policy[:] = self.greedy_epsilon / self.num_actions
		policy[best_action] = 1 - self.greedy_epsilon

		return np.random.choice(actions, p=policy)

	def get_state_tensor(self, state):
		return torch.from_numpy(state.reshape((-1,4))).float()

	def to_variable(input):
		if torch.cuda.is_available():
			return Variable(torch.cat(torch.cuda.FloatTensor(input)))
		else:
			return Variable(torch.cat(torch.FloatTensor(input)))

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.

		# TODO: Model update code comes here and also predicting q for current state
		for episode in range(self.args.eps):
			state = self.get_state_tensor(self.env.reset())
			steps = 0

			while True:
				if self.args.render:
					self.env.render()

				q_values = self.dqnNetwork.policyModel(state)
				action = self.epsilon_greedy_policy(q_values)
				next_state, reward, done, info = self.env.step(action)

				if done:
					reward = -1

				transition = ([state], [action], [reward], [next_state])
				self.replay_memory.append(transition)

				state = next_state
				steps = steps + 1

	def train_q_network(self):
		batch_transitions = self.replay_memory.sample_batch()
		states, actions, rewards, next_states = zip(*batch_transitions)

		states, next_states = self.to_variable(states), self.to_variable(next_states)
		actions, rewards = self.to_variable(actions), self.to_variable(rewards)

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		state = self.env.reset()

		for _ in range(self.replay_memory.burn_in):
			action = self.env.action_space.sample()
			next_state, reward, done, info = self.env.step(action)
			transition = ([state], [action], [reward], [next_state])

			self.replay_memory.append(transition)
			state = next_state

			if done:
				state = self.env.reset()
