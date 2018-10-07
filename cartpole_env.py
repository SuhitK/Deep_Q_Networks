# Test script to check cartpole env fucntionalities
import numpy as np
import gym
import pdb

def main():
    env = gym.make('CartPole-v0')
    # env = gym.wrappers.Monitor(env, '.', force=True)
    state = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            print ("Episode Terminated")
            break
    # env.env.close()
    env.close()

if __name__ == "__main__":
    main()
