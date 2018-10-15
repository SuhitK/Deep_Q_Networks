# CartPole Scripts
# time python main.py --batch_size=64 --lookahead --logger=CartPole-v0_DQN --folder_prefix='Models/New/CartPole-v0_DQN Best/'
# time python main.py --batch_size=64 --double_dqn --logger=CartPole-v0_DDQN --folder_prefix='Models/New/CartPole-v0_DDQN Best/'
# time python main.py --batch_size=64 --duel_dqn --logger=CartPole-v0_DuelDQN --folder_prefix='Models/New/CartPole-v0_DuelDQN Best/'

# MountainCar Scripts
# time python main.py --env=MountainCar-v0 --gamma=1.0 --burn_in=20000 --eps_init=0.85 --target_update=5 --decay --logger=MountainCar-v0_DQN --folder_prefix='Models/New/MountainCar-v0_DQN Best/'
# time python main.py --env=MountainCar-v0 --gamma=1.0 --burn_in=20000 --eps_init=0.85 --target_update=5 --double_dqn --decay --logger=MountainCar-v0_DDQN --folder_prefix='Models/New/MountainCar-v0_DDQN Best/'
# time python main.py --env=MountainCar-v0 --gamma=1.0 --burn_in=20000 --eps_init=0.85 --target_update=5 --duel_dqn --decay --logger=MountainCar-v0_DuelDQN --folder_prefix='Models/New/MountainCar-v0_DuelDQN Best/'