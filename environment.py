import gymnasium as gym  
from gymnasium import spaces
import numpy as np


class CustomTrafficEnv(gym.Env): 
    def __init__(self):
        super(CustomTrafficEnv, self).__init__()
        
      
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        
        self.action_space = spaces.Discrete(3)
        
        
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
      
        super().reset(seed=seed)
        
        scenario = np.random.choice(["leve", "moderado", "pico"])
        if scenario == "leve":
            self.state = np.random.randint(0, 30, size=(4,))
        elif scenario == "moderado":
            self.state = np.random.randint(30, 70, size=(4,))
        else:  
            self.state = np.random.randint(70, 100, size=(4,))
        
        self.done = False
        return self.state, {} 
    

    def step(self, action):
       
        if action == 0: 
            self.state += np.random.randint(-5, 5, size=(4,))
        elif action == 1:  
            self.state += np.random.randint(-10, 0, size=(4,))
        elif action == 2: 
            self.state += np.random.randint(0, 10, size=(4,))
        
        self.state = np.clip(self.state, 0, 100)
        
        
        mean_traffic = np.mean(self.state)
        if mean_traffic > 80:  
            reward = -5
        elif 50 < mean_traffic <= 80:  
            reward = -1
        else:  
            reward = 2
        
       
        done = np.random.random() > 0.95  
        truncated = False  

        return self.state, reward, done, truncated, {}  
        
    def render(self, mode='human'):
        print(f"Estado atual do tráfego: {self.state}")

env = CustomTrafficEnv()
