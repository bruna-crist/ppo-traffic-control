from environment import CustomTrafficEnv

# Teste do ambiente
env = CustomTrafficEnv()
obs, _ = env.reset()  # Reset inicial
print("Estado inicial:", obs)

# Teste básico com ações aleatórias
for _ in range(5):  # Executa 5 passos de interação
    action = env.action_space.sample()  # Seleciona uma ação aleatória
    obs, reward, done, truncated, info = env.step(action)  # Executa a ação
    print(f"Ação: {action}, Novo estado: {obs}, Recompensa: {reward}, Finalizado? {done}, Truncado? {truncated}")
    if done:
        obs, _ = env.reset()  # Reinicia o ambiente se o episódio terminar