from environment import CustomTrafficEnv

env = CustomTrafficEnv()
obs, _ = env.reset() 
print("Estado inicial:", obs)


for _ in range(5): 
    action = env.action_space.sample()  
    obs, reward, done, truncated, info = env.step(action)  
    print(f"Ação: {action}, Novo estado: {obs}, Recompensa: {reward}, Finalizado? {done}, Truncado? {truncated}")
    if done:
        obs, _ = env.reset()  