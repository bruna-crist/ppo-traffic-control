from stable_baselines3 import PPO
from environment import CustomTrafficEnv

env = CustomTrafficEnv()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

model.save("ppo_traffic_control")
