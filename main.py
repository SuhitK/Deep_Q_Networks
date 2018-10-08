import pdb

from DQN.DQN_agent import DQN_Agent

def main():
    agent = DQN_Agent('CartPole-v0', burn_in = 100)
    agent.burn_in_memory()
    train_samples  = agent.replay_memory.sample_batch()
    pdb.set_trace()


if __name__ == "__main__":
    main()
