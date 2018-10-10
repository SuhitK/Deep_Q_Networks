# Deep Reinforcement Lerning Assignment 2
Implementing DQN, Double DQN and Duelling Networks on Atari Environment.

## TO DO
### DQN
- [ ]  model.train() (Use MLP with 3 hidden layers with ReLu activations. Seen this work for the assignment)
- [ ]  model.test()
- [x] Replay memory
- [x] Sample states from gym environment
- [ ] generate performance plots
- [ ] two-step lookahead
- [ ] video capture
- [ ] average total reward of fully trained model for 100 episodes

#### Cartpole Parameters
batch size 32
burn in 10000
lr 1e-4
weight decay 0.01
target update every 50 for first 500 episodes then every 500
no epsilon decay
epsilon init = 0.5
greedy epsilon 0.05

