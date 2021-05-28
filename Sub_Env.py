# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:34:10 2019

@author: Lab User
"""

import numpy as np
#import scipy.io



def create_Env(PaddockLength,NumberOfShepherds,NumberOfSheep,MaximumSheepDistanceToGlobalCentreOfMass):
    
    def cal_dist(v1,v2):    
        distance = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
        return distance
    
    def cal_subgoal_behindcenter(TargetCoordinate, GCM, NumberOfSheep):
      DirectionFromTargetToGlobalCentreOfMass = np.array([GCM[0]-TargetCoordinate[0], GCM[1]-TargetCoordinate[1]])
      NormOfDirectionFromTargetToGlobalCentreOfMass = np.sqrt(DirectionFromTargetToGlobalCentreOfMass[0]**2 + DirectionFromTargetToGlobalCentreOfMass[1]**2)
      NormalisedDirectionFromTargetToGlobalCentreOfMass = DirectionFromTargetToGlobalCentreOfMass / NormOfDirectionFromTargetToGlobalCentreOfMass
      PositionBehindCenterFromTarget = TargetCoordinate + NormalisedDirectionFromTargetToGlobalCentreOfMass * (NormOfDirectionFromTargetToGlobalCentreOfMass + 2 * (NumberOfSheep**(2/3)) + 10)      
      return PositionBehindCenterFromTarget

    # initialize_sheep_mode = {0: np.array([0,1]), 1: np.array([1,0]), 2: np.ones(2)}
    # initialize_shepherd_mode = {0: np.array([0,PaddockLength]), 1: np.array([PaddockLength, 0]), 2: np.array([PaddockLength,PaddockLength])}
    
    """Create the sheep matrix, shepherd matrix and target"""
    TargetCoordinate = np.array([0,0]) # Target coordinates
    # Spawn a random global center of mass near the center of the field    
    SheepMatrix = np.zeros([NumberOfSheep,5]) # initial population of sheep **A matrix of size OBJECTSx5
    
    #create the shepherd matrix
    ShepherdMatrix = np.zeros([NumberOfShepherds,5])  # initial population of shepherds **A matrix of size OBJECTSx5   

    # Initialise Sheep in the upper right corner
    # SheepMatrix[:,0:2] = PaddockLength/4 + np.random.rand(NumberOfSheep,2)*3*PaddockLength/4
    # SheepIndex = 1
    
    # Case 1: Finding driving point: All sheep are collected. Dog is far from driving point
    #Initialise Sheep within MaximumSheepDistanceToGlobalCentreOfMass
    #SheepMatrix[:,[0,1]] = np.random.rand(NumberOfSheep,2)*PaddockLength*1/2 + PaddockLength*1/4
    #r = np.random.randint(3)
    
    '''SheepMatrix[0,[0,1]] = initialize_sheep_mode[r]*PaddockLength/2 + PaddockLength/4
    #SheepMatrix[0,[0,1]] = np.ones(2)*PaddockLength*1/2 + PaddockLength*1/4
    while SheepIndex < NumberOfSheep:
        SheepMatrix[SheepIndex,[0,1]] = SheepMatrix[0,[0,1]] + np.random.uniform(-1,1,2)*MaximumSheepDistanceToGlobalCentreOfMass/2
        GCM = np.sum(SheepMatrix[:,[0,1]],0)/(SheepIndex+1)
        if cal_dist(SheepMatrix[SheepIndex,[0,1]],GCM) <= (MaximumSheepDistanceToGlobalCentreOfMass):
            SheepIndex += 1'''
    
    SheepMatrix[:,[0,1]] = np.random.uniform(PaddockLength*2/3,PaddockLength*2/3+15,[NumberOfSheep,2])
    
    #Initialise Sheep Initial Directions Angle [-pi,pi]
    SheepMatrix[:,2] = np.pi - np.random.rand(len(SheepMatrix[:,2]))*2*np.pi #1 - because just having one column
    #Add the index of each sheep into the matrix
    SheepMatrix[:,4] = np.arange(0,len(SheepMatrix[:,4]),1)
    # Initialise Shepherd in the lower left corner
    ShepherdMatrix[:,0:2] = TargetCoordinate
    #ShepherdMatrix[:,0:2] = cal_subgoal_behindcenter(TargetCoordinate, GCM, NumberOfSheep)
    #ShepherdMatrix[:,0:2] = np.array([PaddockLength,PaddockLength])
    #Initialise Sheep Initial Directions Angle [-pi,pi]
    SheepMatrix[:,2] = np.pi - np.random.randn(len(SheepMatrix[:,2]))*2*np.pi #1 - because just having one column

    #Add the index of each sheep into the matrix
    SheepMatrix[:,4] = np.arange(0,len(SheepMatrix[:,4]),1)
    #Initialise Shepherds Initial Directions Angle [-pi,pi]
    ShepherdMatrix[:,2]= np.pi - np.random.randn(NumberOfShepherds)*2*np.pi

    #Add the index of each shepherd into the matrix
    ShepherdMatrix[:,4] = np.arange(0,len(ShepherdMatrix[:,4]),1)
      
    return(SheepMatrix,ShepherdMatrix,TargetCoordinate)