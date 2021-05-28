# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:23:39 2019

@author: Lab User
"""

#import keras
from DQN_Agent_Keras import DQN_Agent_Keras as DQN
from Environment import Environment
import datetime
import pandas as pd
import numpy as np
import pickle
import copy

from tensorflow.keras.models import load_model


N_EPISODE = 15000
MAX_STEP = 1000
SAVE_NETWORK = 100 # 100 episodes

NumberOfSheep = 30
NumberOfShepherds = 1
# Obstacles = None
obs_train = 'Obstacles1'
obs_test = 'Obstacles3a'
Obstacles = np.load(obs_test + '.npy')

# Initialize network and memory
env = Environment(NumberOfSheep,NumberOfShepherds,Obstacles)
dqn_agent = DQN(state_size = 4, 
                action_space = 5)

dqn_agent.model = load_model('Original_Models/DQNmodel_driving_obs1_20210420-0325_ep50000.h5')
dqn_agent.target_model = load_model('Original_Models/DQNmodel_driving_obs1_20210420-0325_ep50000.h5')
#Create header for the saving DQN learning file

header = ["Ep","Step","Total_reward","Epsilon","Termination_Code","Episode_Time"]
filename = "Data/Retrain_" + obs_train + "_to_" + obs_test + ".csv"

with open(filename, 'w') as f:
    pd.DataFrame(columns = header).to_csv(f,encoding='utf-8', index=False, header = True)
    
t0 = datetime.datetime.now()
for episode_i in range(N_EPISODE):
    
    terminate, s = env.reset()
    total_reward = 0
    LearningState = None
    for step in range(MAX_STEP):
        
        if env.AreFurthestSheepCollected == 1:
            LearningState = copy.deepcopy(s)
            a = dqn_agent.act(s)       
            s2, r, terminate = env.step(Action=a)
        else:
            s2, r, terminate = env.step(Action=None)
                   
        # View environment
        # env.view()
        if LearningState is not None:
            if (env.AreFurthestSheepCollected == 1) or np.abs(terminate)== 1:
                # Experience replay
                dqn_agent.step(LearningState, a, r, s2, np.abs(terminate))                               
                total_reward += r
        
        s = s2
                                        
        if (np.abs(terminate)):
            break
        
    # iteration to save the network architecture and weights
    if(np.mod(episode_i+1, SAVE_NETWORK) == 0):
        dqn_agent.target_train() # Replace the learning weights for target model with soft replacement
        now = datetime.datetime.now()
        dqn_agent.save_model("Retraining_Models/DQNmodel_retrain_" + obs_train + "_to_" + obs_test + "_ep" + str(episode_i+1) + ".h5")
    
    t1 = datetime.datetime.now()         
    print('Episode %d ends. Number of steps is: %d. Total reward: %.2f. Epsilon = %.3f . Termination code: %d.' % (episode_i+1, step+1, total_reward, dqn_agent.epsilon, terminate))
    print('Episode time elapsed = ' + str((t1-t0).total_seconds()))
    
    #Saving data to file
    save_data = np.hstack([episode_i+1,step+1,total_reward,dqn_agent.epsilon,terminate,(t1-t0).total_seconds()]).reshape(1,6)
    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(f,encoding='utf-8', index=False, header = False)

    #update epsilon
    dqn_agent.update_epsilon()
    t0 = t1

# save final model    
dqn_agent.save_model("Retraining_Models/DQNmodel_retrain_" + obs_train + "_to_" + obs_test + "_ep" + str(episode_i+1) + ".h5")

M = {}

for index in dqn_agent.memory.memory:
    M[index] = dqn_agent.memory.memory[index]._asdict()
    
with open('Data/memory_retrain_' + obs_train + "_to_" + obs_test + '.pkl', 'wb') as output:
    pickle.dump(M, output, pickle.HIGHEST_PROTOCOL)
