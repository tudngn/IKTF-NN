# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:49:30 2019

@author: TALabUser
"""

#import keras
#from tensorflow.keras.models import load_model
from Environment import Environment
import datetime
import pandas as pd
import numpy as np

N_EPISODE = 100
MAX_STEP = 5000

NumberOfSheep = 50
NumberOfShepherds = 1
#Obstacles = None
Obstacles = np.load('Obstacles3.npy')
# Initialize network and memory
# driving_model = load_model('Model_50000EP_09999decay_min005/DQNmodel_driving_20201222-1548_ep50000.h5')
env = Environment(NumberOfSheep,NumberOfShepherds,Obstacles)

#Create header for the saving DQN learning file
'''now = datetime.datetime.now()
header = ["Ep","Step","Termination_Code"]
filename = "Data/Testing_" + now.strftime("%Y%m%d-%H%M") + ".csv"

with open(filename, 'w') as f:
    pd.DataFrame(columns = header).to_csv(f,encoding='utf-8', index=False, header = True)'''

for episode_i in range(0,N_EPISODE):
    
    terminate, s = env.reset()
    
    for step in range(0,MAX_STEP):
        #print(step)
        #a = np.argmax(driving_model.predict(s.reshape(1,len(s))))            
        s2, _, terminate = env.step(Action=None)  
                           
        # View environment
        env.view()
        s = s2
                                        
        if (np.abs(terminate)):
            break
    
    #Saving data to file
    '''save_data = np.hstack([episode_i+1,step+1,terminate]).reshape(1,3)
    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(f,encoding='utf-8', index=False, header = False)'''
                
    #update status
    print('Episode %d ends. Number of steps is: %d. Termination code: %d' % (episode_i+1, step+1, terminate))
