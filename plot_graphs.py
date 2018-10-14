import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    training_reward = np.loadtxt('RewardsCSV/Training_Rewards_MountainCar-v0_01:23:30.105578.csv')
    average_reward = np.loadtxt('RewardsCSV/Average_Rewards_MountainCar-v0_01:23:30.105578.csv')
    x_axis = np.linspace(100,10000, num = 100)

    plt.plot(x_axis,training_reward)
    plt.title('Average Training Reward for Double DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.show()



if __name__ =='__main__':
    main()
