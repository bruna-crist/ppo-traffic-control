import csv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import CustomTrafficEnv
# from test_environment import CustomTrafficEnv

print("Carregando modelo treinado...")
model = PPO.load("ppo_traffic_control")
print("Modelo carregado com sucesso!")


# Criar o ambiente
env = CustomTrafficEnv()

# Simulação de recompensas
rewards = []
obs, _ = env.reset()  # Reset inicial
print("Iniciando a avaliação do modelo...")

for step in range(100):  # Avaliação em 100 passos
    action, _ = model.predict(obs)  # Predição da ação
    obs, reward, done, truncated, info = env.step(action)  # Executa a ação
    rewards.append(reward)  # Salva a recompensa

    # Exibe logs detalhados no terminal
    print(f"Passo: {step + 1}")
    print(f"Ação escolhida: {action}")
    print(f"Estado atual: {obs}")
    print(f"Recompensa: {reward}")
    print(f"Episódio finalizado? {done}")
    print("-" * 40)

    if done:
        obs, _ = env.reset()  # Reinicia o ambiente ao término do episódio

#salavar resultados
with open("rewards.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Step", "Reward"])
    for i, reward in enumerate(rewards):
        writer.writerow([i, reward])
print("Recompensas salvas em 'rewards.csv'")

# # Gráfico de recompensas
plt.plot(rewards)
plt.title("Recompensas ao longo dos passos")
plt.xlabel("Passos")
plt.ylabel("Recompensa")
plt.show()

