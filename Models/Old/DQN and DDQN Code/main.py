import os
import pdb
import shutil
import argparse

from DQN.DQN_agent import DQN_Agent


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
	parser.add_argument('--render', dest='render', type=bool, default=False)
	parser.add_argument('--train', dest='train', type=int, default=1)
	parser.add_argument('--model', dest='model_file', type=str)
	parser.add_argument('--gamma', dest='gamma', type=int, default=0.9)
	parser.add_argument('--burn_in', dest='burn_in', type=int, default=10000)
	parser.add_argument('--batch_size', dest='bsz', type=int, default=32)
	parser.add_argument('--episodes', dest='epi', type=int, default=10000)
	parser.add_argument('--test_episodes', dest='test_epi', type=int, default=100)
	parser.add_argument('--save_episodes', dest='save_epi', type=int, default=1000)
	parser.add_argument('--eps_init', dest='epsilon_init', type=int, default=0.5)
	parser.add_argument('--eps_stop', dest='epsilon_stop', type=int, default=0.05)
	parser.add_argument('--eps_iter', dest='epsilon_iter', type=int, default=100000)
	parser.add_argument('--eps_greedy', dest='greedy_epsilon', type=int, default=0.05)
	parser.add_argument('--reset_dir', dest='reset_dir', type=bool, default=True)
	parser.add_argument('--double_dqn', dest='double_dqn', action='store_true')
	return parser.parse_args()


def print_user_flags(user_flags, line_limit=80):
	print("-" * 80)

	for flag_name in vars(user_flags):
		value = "{}".format(getattr(user_flags, flag_name))
		log_string = flag_name
		log_string += "." * (line_limit - len(flag_name) - len(value))
		log_string += value
		print(log_string)


def main():
	args = parse_arguments()
	print_user_flags(args)

	if not os.path.exists('PolicyModel/'):
		os.makedirs('PolicyModel/')
	elif args.reset_dir:
		shutil.rmtree('PolicyModel/', ignore_errors=True)
		os.makedirs('PolicyModel/')
	if not os.path.exists('TargetModel/'):
		os.makedirs('TargetModel/')
	elif args.reset_dir:
		shutil.rmtree('TargetModel/', ignore_errors=True)
		os.makedirs('TargetModel/')
	if not os.path.exists('RewardsCSV/'):
		os.makedirs('RewardsCSV/')
	elif args.reset_dir:
		shutil.rmtree('RewardsCSV/', ignore_errors=True)
		os.makedirs('RewardsCSV/')

	agent = DQN_Agent(args, memory_size=100000, burn_in=args.burn_in)
	agent.burn_in_memory()
	agent.train()
	agent.agent_close()


if __name__ == "__main__":
	main()