from stable_baselines3 import PPO
from environment import CustomTrafficEnv

# Cria o ambiente
env = CustomTrafficEnv()

# Configura o modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Treina o modelo
model.learn(total_timesteps=10000)

# Salva o modelo
model.save("ppo_traffic_control")
