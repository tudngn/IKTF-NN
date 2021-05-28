# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:19:36 2019

@author: Lab User
"""
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from Memory import ReplayBuffer


BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 32             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 5e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate 
EPSILON = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
UPDATE_NN_EVERY = 5        # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 1000          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 5000     # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)


# Deep Q Network off-policy
class DQN_Agent_Keras: 
   
    def __init__(
            self,
            state_size,
            action_space,
            compute_weights = False
    ):
        self.state_size = state_size
        self.action_space = action_space
        self.compute_weights = compute_weights
        self.epsilon = EPSILON
        
        #Initialize networks
        self.model        = self.create_model()
        self.target_model = self.create_model()
              
        # Initialize replay memory
        self.memory = ReplayBuffer(1, 
                                   BUFFER_SIZE, 
                                   BATCH_SIZE, 
                                   EXPERIENCES_PER_SAMPLING, 
                                   compute_weights)
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0      

      
    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dense(self.action_space))
        model.add(Activation('linear'))    
        adam = Adam(lr=LR)
        model.compile(optimizer = adam,
                  loss='mse')
        return model
  

    def step(self, state, action, reward, next_state, done, regulator=1.0):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, regulator)
        
        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > EXPERIENCES_PER_SAMPLING:
                sampling = self.memory.sample()
                self.learn(sampling)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()
            
    
    def act(self, state):
        
        #a_max = np.argmax(self.model.predict(state.reshape(1,len(state))))
      
        if (np.random.rand() < self.epsilon):
            a_chosen = np.random.randint(self.action_space)
        else:
            a_chosen = np.argmax(self.model.predict(state.reshape(1,len(state))))
      
        return a_chosen
    
    
    def learn(self, sampling):
        states, actions, rewards, next_states, dones, weights, indices  = sampling
        targets = self.target_model.predict(states)
        delta = np.zeros([BATCH_SIZE,1])
      
        for i in range(BATCH_SIZE):
            action = actions[i,0]
            reward = rewards[i,0]
            next_state = next_states[i,:]
            done = dones[i,0]
        
            # if terminated, only equals reward
            delta[i,0] = targets[i,action]
            if done:
                targets[i,action] = reward
            else:
                Q_future = np.max(self.target_model.predict(next_state.reshape(1,len(next_state))))
                targets[i,action] = reward + Q_future * GAMMA
            
            delta[i,0] = abs(targets[i,action] - delta[i,0])    
        
        # ------------------- train local network ------------------- # 
        self.model.train_on_batch(states, targets)
        # ------------------- update target network ------------------- #
        self.target_train()
        # ------------------- update priorities ------------------- #
        self.memory.update_priorities(delta, indices)
        
          
    def target_train(self): 
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(0, len(target_weights)):
            target_weights[i] = weights[i]*TAU + target_weights[i] * (1 - TAU)
      
        self.target_model.set_weights(target_weights) 
    
    
    def update_epsilon(self):
        self.epsilon =  max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    
    def save_model(self,model_path):
        self.model.save(model_path)

 