
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import CustomTrafficEnv

# Carrega o modelo treinado em 2024
model = PPO.load("ppo_traffic_control.zip")

NUM_EXECUCOES = 30
PASSOS = 100


def avaliar(tipo_agente, seed):
    np.random.seed(seed)
    env = CustomTrafficEnv()
    obs, _ = env.reset()

    recompensa_total = 0

    for _ in range(PASSOS):
        if tipo_agente == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif tipo_agente == "fixo":
            action = 1
        else:
            action = np.random.randint(0, 3)

        obs, reward, done, truncated, _ = env.step(int(action))
        recompensa_total += reward

        if done or truncated:
            obs, _ = env.reset()

    env.close()
    return recompensa_total


resultados = {
    "PPO": [],
    "Aleatório": [],
    "Regra fixa": []
}

for i in range(NUM_EXECUCOES):
    seed = 2024 + i

    resultados["PPO"].append(avaliar("ppo", seed))
    resultados["Aleatório"].append(avaliar("aleatorio", seed))
    resultados["Regra fixa"].append(avaliar("fixo", seed))

for nome, valores in resultados.items():
    print(f"{nome}: média = {np.mean(valores):.2f}")

plt.figure(figsize=(8, 5))
plt.boxplot(list(resultados.values()),
            labels=list(resultados.keys()))

plt.title("Comparação de estratégias de controle")
plt.ylabel("Recompensa acumulada (100 passos)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("comparacao_tres_agentes.png", dpi=150)
plt.show()