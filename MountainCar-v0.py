import gymnasium as gym
import numpy as np
import random
import pygame
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Buffer lebih besar
        self.gamma = 0.99  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # Eksplorasi lebih lama
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Inisialisasi pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 64

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(200):
            frame = env.render()
            pygame.surfarray.blit_array(screen, np.transpose(frame, (1, 0, 2)))
            pygame.display.flip()

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 🔹 **REWARD SHAPING**
            if terminated:
                reward = 100  # Reward besar jika mencapai puncak
            else:
                reward += 10 * abs(next_state[1])  # Bonus untuk kecepatan
                reward -= 0.1  # Penalti kecil untuk mencegah diam di tempat
            
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Langkah: {time}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
                break
        
        agent.replay(batch_size)
        clock.tick(30)  # Batasi FPS untuk rendering lebih halus

    pygame.quit()
    env.close()
