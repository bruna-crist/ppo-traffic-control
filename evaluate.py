import csv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import CustomTrafficEnv


print("Carregando modelo treinado...")
model = PPO.load("ppo_traffic_control")
print("Modelo carregado com sucesso!")



env = CustomTrafficEnv()


rewards = []
obs, _ = env.reset()  
print("Iniciando a avaliação do modelo...")

for step in range(100): 
    action, _ = model.predict(obs) 
    obs, reward, done, truncated, info = env.step(action)  
    rewards.append(reward)  

    print(f"Passo: {step + 1}")
    print(f"Ação escolhida: {action}")
    print(f"Estado atual: {obs}")
    print(f"Recompensa: {reward}")
    print(f"Episódio finalizado? {done}")
    print("-" * 40)

    if done:
        obs, _ = env.reset() 

with open("rewards.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Step", "Reward"])
    for i, reward in enumerate(rewards):
        writer.writerow([i, reward])
print("Recompensas salvas em 'rewards.csv'")

plt.plot(rewards)
plt.title("Recompensas ao longo dos passos")
plt.xlabel("Passos")
plt.ylabel("Recompensa")
plt.show()

