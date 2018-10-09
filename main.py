import pdb

from DQN.DQN_agent import DQN_Agent


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
	parser.add_argument('--render', dest='render', type=bool, default=False)
	parser.add_argument('--train', dest='train', type=int, default=1)
	parser.add_argument('--model', dest='model_file', type=str)
	parser.add_argument('--episodes', dest='eps', type=int, default=2000)
	return parser.parse_args()


def main():
	args = parse_arguments()

    agent = DQN_Agent(args, burn_in=100)
    agent.burn_in_memory()
    train_samples  = agent.replay_memory.sample_batch()
    pdb.set_trace()


if __name__ == "__main__":
    main()
