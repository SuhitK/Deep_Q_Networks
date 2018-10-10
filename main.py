import pdb
import argparse

from DQN.DQN_agent import DQN_Agent


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
	parser.add_argument('--render', dest='render', type=bool, default=False)
	parser.add_argument('--train', dest='train', type=int, default=1)
	parser.add_argument('--model', dest='model_file', type=str)
	parser.add_argument('--gamma', dest='gamma', type=int, default=0.9)
	parser.add_argument('--batch_size', dest='bsz', type=int, default=32)
	parser.add_argument('--episodes', dest='epi', type=int, default=2000)
	parser.add_argument('--test_episodes', dest='test_epi', type=int, default=100)
	parser.add_argument('--eps_init', dest='epsilon_init', type=int, default=0.5)
	parser.add_argument('--eps_stop', dest='epsilon_stop', type=int, default=0.05)
	parser.add_argument('--eps_iter', dest='epsilon_iter', type=int, default=10000)
	parser.add_argument('--eps_greedy', dest='greedy_epsilon', type=int, default=0.05)
	return parser.parse_args()


def main():
	args = parse_arguments()

	agent = DQN_Agent(args, burn_in=100)
	agent.burn_in_memory()
	# train_samples = agent.replay_memory.sample_batch()
	# pdb.set_trace()

	agent.train()


if __name__ == "__main__":
	main()
