from DQN.DQN_agent import DQN_Agent

def main():
    agent = DQN_Agent('CartPole-v0', burn_in = 100)
    agent.burn_in_memory()

if __name__ == "__main__":
    main()
