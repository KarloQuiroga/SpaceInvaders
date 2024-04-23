# Instalamos las librerías necesarias.
# !pip install tensorflow==2.3.1 gym keras-rl2 gym[atari, accept-rom-license]==0.21.0 pyglet==1.2.4

import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Crear el ambiente
env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# Definición del modelo
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(height, width, channels)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(height, width, channels, actions)
model.summary()

# Construcción del agente DQN
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=100000, window_length=3)
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg', nb_steps_warmup=1000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))

# Ejecución del modelo
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice([0, 1, 2, 3, 4, 5])
        n_state, reward, done, info = env.step(action)
        score += reward
        print('Episode: {} Score: {}'.format(episode, score))
    env.close()

