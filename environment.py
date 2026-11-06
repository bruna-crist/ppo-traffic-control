import gymnasium as gym  # gymnasium
from gymnasium import spaces
import numpy as np


class CustomTrafficEnv(gym.Env): #Classe criada que herda de gym.env para criar o ambiente
    def __init__(self):
        super(CustomTrafficEnv, self).__init__()
        
        # Espaço de observação: Simula estado do tráfego (ex.: fluxo nas faixas)
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        # Espaço de ação:  3 ações (0 = manter, 1 = aumentar, 2 = diminuir tempo de semáforo)
        self.action_space = spaces.Discrete(3)
        
        # Variáveis iniciais
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        # Configuração seed 
        super().reset(seed=seed)
        # Define o estado inicial com base em cenários variados, escolha aleátoria de cenários
        scenario = np.random.choice(["leve", "moderado", "pico"])
        if scenario == "leve":
            self.state = np.random.randint(0, 30, size=(4,))
        elif scenario == "moderado":
            self.state = np.random.randint(30, 70, size=(4,))
        else:  # Cenário de pico
            self.state = np.random.randint(70, 100, size=(4,))
        
        self.done = False
        return self.state, {} # Retorna o estado inicial e um dicionário vazio para info
    

    def step(self, action):
        # Aplica a ação e atualiza o estado
        if action == 0:  # Manter
            self.state += np.random.randint(-5, 5, size=(4,))
        elif action == 1:  # Aumentar
            self.state += np.random.randint(-10, 0, size=(4,))
        elif action == 2:  # Diminuir
            self.state += np.random.randint(0, 10, size=(4,))
        
        self.state = np.clip(self.state, 0, 100)
        
        # Calcula a recompensa
        mean_traffic = np.mean(self.state)
        if mean_traffic > 80:  # Congestionamento severo
            reward = -5
        elif 50 < mean_traffic <= 80:  # Congestionamento moderado
            reward = -1
        else:  # Fluxo livre
            reward = 2
        
       # Define a terminação e o truncamento do episódio
        done = np.random.random() > 0.95  # Simulação de fim aleatório
        truncated = False  # Não estamos tratando truncamento, então sempre False

        return self.state, reward, done, truncated, {}  # Retorna 5 valores
        
    def render(self, mode='human'):
        # Exibe o estado atual do tráfego
        print(f"Estado atual do tráfego: {self.state}")

#criar o ambiente
env = CustomTrafficEnv()
