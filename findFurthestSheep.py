# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:54:22 2019

@author: z5095790
"""

import numpy as np

def findFurthestSheep(SheepMatrix,SheepGlobalCentreOfMass,TargetCoordinate,NumberOfShepherds,MaximumSheepDistanceToGlobalCentreOfMass):
    
    def normalize(v):
        if v[0]==0 and v[1]==0:
            return [0,0]
        else:
            return v / np.linalg.norm(v)
        
        
    def v_len(v):
        return np.sqrt(v[0]**2 + v[1]**2)
   
    def angle_vector(u,v):
        return np.arccos(np.dot(u,v)/(v_len(u)*v_len(v)))
    
    
    ''' INPUTS

    % SheepMatrix               = Sheep position matrix
    % NumberOfShepherds         = Number of shepherds
    
    % OUTPUT
    % IndexOfFurthestSheep     = Index of Furthest Sheep matrix with the number of selected sheep equal to the number of shepherd
    % AreFurthestSheepCollected to identify if there are still outlier sheep'''
    NumberOfSheep = len(SheepMatrix)
    SheepDistanceToGlobalCentreOfMass = np.sqrt((SheepMatrix[:,0]-SheepGlobalCentreOfMass[0])**2+(SheepMatrix[:,1]-SheepGlobalCentreOfMass[1])**2)
    
    # Find and sort the distance between LCM of the sheep and every sheep
    GCMDistanceToSheepWithIndex = np.zeros([NumberOfSheep,2])
    GCMDistanceToSheepWithIndex[:,0] = SheepDistanceToGlobalCentreOfMass
    GCMDistanceToSheepWithIndex[:,1] = SheepMatrix[:,4]
    SortedGCMDistanceToSheepWithIndex = GCMDistanceToSheepWithIndex[GCMDistanceToSheepWithIndex[:,0].argsort()[::-1]]
    # Find the index of the furthest sheep to LCM
    IndexOfFurthestSheep = None
    if NumberOfShepherds > 1:
        IndexOfFurthestSheep = SortedGCMDistanceToSheepWithIndex[0:NumberOfShepherds,1].astype(int)
    else:
        IndexOfFurthestSheep = SortedGCMDistanceToSheepWithIndex[0,1].astype(int)
        '''v_CM_Target = [TargetCoordinate[0] - SheepGlobalCentreOfMass[0], TargetCoordinate[1] - SheepGlobalCentreOfMass[1]]
        for i in range(len(SortedGCMDistanceToSheepWithIndex)):
            temp_id = SortedGCMDistanceToSheepWithIndex[i,1].astype(int)
            v_Sheep_CM = [SheepGlobalCentreOfMass[0] - SheepMatrix[temp_id,0], SheepGlobalCentreOfMass[1] - SheepMatrix[temp_id,1]]
            if angle_vector(v_Sheep_CM,v_CM_Target) <= 3*np.pi/4:
                IndexOfFurthestSheep = temp_id
                break'''
    # find the collection status of the furthest sheep
    AreFurthestSheepCollected = 0
    
    if SortedGCMDistanceToSheepWithIndex[0,0] <= MaximumSheepDistanceToGlobalCentreOfMass:
        AreFurthestSheepCollected = 1
    
    return(IndexOfFurthestSheep,AreFurthestSheepCollected)