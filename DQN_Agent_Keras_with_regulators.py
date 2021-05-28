# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:19:36 2019

@author: Lab User
"""
import numpy as np
import math
import cdd
import copy
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

MIN_DISTANCE = 5

# Deep Q Network off-policy
class DQN_Agent_Keras: 
   
    def __init__(
            self,
            state_size,
            action_space,
            rule_list,
            is_weak_rule,
            env_size,
            compute_weights = False
    ):
        self.state_size = state_size
        self.action_space = action_space
        self.rule_list = rule_list
        self.is_weak_rule = is_weak_rule
        self.env_size = env_size
        self.compute_weights = compute_weights
        self.max_epsilon = EPSILON
        
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
        
        # Initialize epsilon array:
        self.num_rules = len(rule_list)
        # extracting centre of polytopes
        self.find_centroid_polytopes()
        self.create_epsilon_array()
       


    ''' Convert the rule to H-representation'''
    def convert_to_H_representation(self, rule):
        H_rep = np.hstack([np.reshape(rule[:,-2], [len(rule),1]), rule[:,0:-2]])
        return H_rep
    
    
    ''' Convert the rule to V-representation'''
    def convert_to_V_representation(self,H_rep):
        mat = copy.deepcopy(H_rep)
        mat_dd = cdd.Matrix(mat)
        mat_dd.rep_type = cdd.RepType.INEQUALITY
        try:
            poly = cdd.Polyhedron(mat_dd)
        except:
            mat = np.around(mat, decimals = 7) 
            mat_dd = cdd.Matrix(mat)
            mat_dd.rep_type = cdd.RepType.INEQUALITY
            poly = cdd.Polyhedron(mat_dd)
        #print(poly)
        ext = poly.get_generators()
        #print(ext)
        v = np.array(ext)
        N = len(v)
        if N == 0:
            vertices = None
        else:
            true_indexes = np.zeros(N).astype(bool)
            for r in range(N):
                if v[r,0] == 1:
                    true_indexes[r] = True
            vertices = v[true_indexes, 1:]
            if len(vertices) < 3:
                vertices = None
                
        return vertices


    def distance(self, a, b):
        return np.abs(a-b)
    
    
    def full_scale_state(self, state):
        full_state = np.zeros(4)
        full_state[0] = state[0]*self.env_size*np.sqrt(2)
        full_state[1] = state[1]*180/np.pi
        full_state[2] = state[2]*self.env_size*np.sqrt(2)
        full_state[3] = state[3]*180/np.pi
        return full_state
    

    '''def check_rule_truth_value(self, rule, point):
    
        truth_value = False
        values = np.matmul(rule[:,:-2], point) + rule[:,-2]        
        if np.prod(values > 0-1e-5) == 1: # soft margin: 0-1e-5
            truth_value = True
    
        return(truth_value)'''

    
    '''def which_rule_activate(self, state):
        index = None
        for i in range(self.num_rules):
            rule = self.rule_list[i]
            if self.check_rule_truth_value(rule, state):
                index = i
                break
            
        return index'''


    def find_centroid_polytopes(self):
        #self.centroid_of_polytopes = []
        self.centroid_of_weak_polytopes = []
        for i in range(self.num_rules):
            rule = self.rule_list[i]
            H_rep = self.convert_to_H_representation(rule)
            vertices = self.convert_to_V_representation(H_rep)
            centroid = None
            if vertices is not None:
                centroid = np.mean(vertices, axis=0)
               
            #self.centroid_of_polytopes.append(centroid)
            if self.is_weak_rule[i] and centroid is not None:
                self.centroid_of_weak_polytopes.append(centroid)


    def distance_to_closest_weak_polytopes(self, full_state):
        
        point = full_state[2] # extract only 2 dimensions related to distance
        d = np.inf
        for i in range(len(self.centroid_of_weak_polytopes)):
            tmp = self.distance(point, self.centroid_of_weak_polytopes[i][2])
            if tmp < d:
                d = tmp
                
        return d
                
      
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


    def create_epsilon_array(self):
        # create an dictionary whose keys are the ranges with size = MIN_DISTANCE
        # Each range has a difference value of epsilon
        self.epsilon_dict = {}
        for i in range(1, int(np.sqrt(2)*self.env_size/MIN_DISTANCE)+1):
            d = i*MIN_DISTANCE
            self.epsilon_dict[i] = max(np.exp(-2*d/(np.sqrt(2)*self.env_size - d)), 0.05)

                
    '''def create_priority_array(self):
        self.priority_array = np.ones([self.num_rules]) 
        for i in range(self.num_rules):
            d = self.distance_to_closest_weak_polytopes(i)
            if d is not None:
                if d > MIN_DISTANCE:
                    if d < np.sqrt(2)*self.env_size:
                        self.priority_array[i] = np.exp(-d/(np.sqrt(2)*self.env_size - d))
                    else:
                        self.priority_array[i] = 0.1
            if self.priority_array[i] < 0.1:
                self.priority_array[i] = 0.1'''


    def compute_epsilon(self, d):
        epsilon = self.max_epsilon
        if d > MIN_DISTANCE:
            if d <= MIN_DISTANCE*max(self.epsilon_dict):
                epsilon = self.epsilon_dict[int(d // MIN_DISTANCE)]
            else:
                epsilon = 0.05
        if epsilon < 0.05:
            epsilon = 0.05
            
        return epsilon


    def compute_priority_regulator(self, d):
        priority_regulator = 1
        if d > MIN_DISTANCE:
            if d < np.sqrt(2)*self.env_size:
                priority_regulator = np.exp(-d/(np.sqrt(2)*self.env_size - d))
            else:
                priority_regulator = 0.1
        if priority_regulator < 0.1:
            priority_regulator = 0.1
            
        return priority_regulator
    

    def step(self, state, action, reward, next_state, done, distance_to_weak_polytopes = 0):
        # Save experience in replay memory
        priority_regulator = self.compute_priority_regulator(distance_to_weak_polytopes)        
        self.memory.add(state, action, reward, next_state, done, priority_regulator)
        
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
            
    
    def act(self, state, distance_to_weak_polytopes = 0):
        epsilon = self.compute_epsilon(distance_to_weak_polytopes)      
        if (np.random.rand() < epsilon):
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
        for i in self.epsilon_dict:
            self.epsilon_dict[i] = max(EPSILON_MIN, self.epsilon_dict[i] * EPSILON_DECAY)
        self.max_epsilon = max(EPSILON_MIN, self.max_epsilon * EPSILON_DECAY)
    
    
    def save_model(self,model_path):
        self.model.save(model_path)

 