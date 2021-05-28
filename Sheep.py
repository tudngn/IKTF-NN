# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:28:27 2019

@author: Lab User
"""
import numpy as np

class Sheep:
    
    ''' INPUTS
    # SheepPosition=Sheep position matrix
    # NeighbourhoodRange= The radius to Identify Local Centre of Mass
    # ShepherdPosition=Position matrix of shepherds
    # InfluenceDistance=Radius within it the shepherd can influence the sheep
    # MinDistanceShepherdSheep = The minimum distance that shepherd can approach a sheep
    # SheepStep = Speed of Sheep
    # ShepherdStep = Speed of Shepherd
    # Obstacles= The matrix include the positions and size of the obstacles
    # AvoidanceRadius=The radius within it the sheep reacts to the obstacles
    
    # OUTPUTS
    # PreviousDirection=Direction of movement from previous step
    # RepulsionSheepDirection=Direction of repulsion from other sheep
    # ShepherdRepulsionDirection=Direction of repulsion from shepherds
    # AttractionDirection=Direction of attraction to sheep' LCM.
    # AvoidanceForce=Direction of Obstacle Avoidance.
    '''
    
    def __init__(
            self,
            PaddockLength,
            NumberOfSheep,
            NumberOfShepherds,
            SheepRepulsionRange,
            NeighbourhoodRange,
            InfluenceDistance,
            MinDistanceShepherdSheep,
            SheepStep,
            ShepherdStep,
            Obstacles,
            AvoidanceRadius,
            WeightOfInertia,
            WeightRepellFromOtherSheep,
            WeightRepellFromShepherd,
            WeightAttractionToLCM,
            AvoidanceWeight,
            NoiseLevel
            ):

        self.PaddockLength = PaddockLength
        self.NumberOfSheep = NumberOfSheep
        self.NumberOfShepherds = NumberOfShepherds
        self.SheepRepulsionRange = SheepRepulsionRange
        self.NeighbourhoodRange = NeighbourhoodRange
        self.InfluenceDistance = InfluenceDistance
        self.MinDistanceShepherdSheep = MinDistanceShepherdSheep
        self.SheepStep = SheepStep
        self.ShepherdStep = ShepherdStep
        self.Obstacles = Obstacles
        self.AvoidanceRadius = AvoidanceRadius
        self.WeightOfInertia = WeightOfInertia
        self.WeightRepellFromOtherSheep = WeightRepellFromOtherSheep
        self.WeightRepellFromShepherd = WeightRepellFromShepherd
        self.WeightAttractionToLCM = WeightAttractionToLCM
        self.AvoidanceWeight = AvoidanceWeight
        self.NoiseLevel = NoiseLevel
        
        self.SheepMatrix = None
        self.ShepherdMatrix = None


    def v_len(self, v):
        return np.sqrt(v[0]**2 + v[1]**2)
    
    
    def dist_p2p(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    
    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v/norm
        else:
            return np.array([0,0])

        
    def angle_vectors(self, u,v):
        return np.arccos(np.dot(u,v)/(self.v_len(u)*self.v_len(v)))
    
    
    def perpendicular_vector_pair(self, v):
        if v[1] == 0:
            nv1 = [0,-1]
            nv2 = [0,1]
        else:
            v1 = [1, -v[0]/v[1]]
            v2 = [-1, v[0]/v[1]]
            nv1 = self.normalize(v1)
            nv2 = self.normalize(v2)
        return (nv1, nv2)
    
    
    def compute_sheep_repulsion_regulator(self, distance):
        M_r = self.SheepStep*np.exp(-2*distance/self.SheepRepulsionRange)
        return M_r


    def compute_shepherd_repulsion_regulator(self, distance):
        M_e = self.ShepherdStep/self.SheepStep*np.exp(-2*distance/(self.InfluenceDistance-self.MinDistanceShepherdSheep))
        return M_e

    
    def compute_obstacle_avoidance_regulator(self, distance):
        M_a = self.SheepStep*np.exp(-2*distance/self.AvoidanceRadius)
        return M_a

    
    def compute_previous_direction(self, SheepIndex):
        PreviousDirection = np.array([np.cos(self.SheepMatrix[SheepIndex,2]),np.sin(self.SheepMatrix[SheepIndex,2])])
        return PreviousDirection
    
    
    def compute_sheep_repulsion(self, SheepIndex, sheepPosition):
        # Determine number of sheep within the sheep-sheep repulsion range
        sheep_repulsion = np.array([0,0])
        for j in range(self.NumberOfSheep):        
            if (j != SheepIndex):
                distance = self.dist_p2p(sheepPosition, self.SheepMatrix[j,0:2])
                if distance <= self.SheepRepulsionRange:
                    force = self.normalize(sheepPosition - self.SheepMatrix[j,0:2])
                    M_r = self.compute_sheep_repulsion_regulator(distance)
                    sheep_repulsion = sheep_repulsion + M_r*force
            
        return self.normalize(sheep_repulsion)   
    

    def compute_shepherd_repulsion(self, sheepPosition):
        # Determine number of shepherds within the shepherd-sheep influence range
        shepherd_repulsion = np.array([0,0])
        for k in range(self.NumberOfShepherds):        
            distance = self.dist_p2p(sheepPosition, self.ShepherdMatrix[k,0:2])
            if distance <= self.InfluenceDistance:
                force = self.normalize(sheepPosition - self.ShepherdMatrix[k,0:2])
                M_e = self.compute_shepherd_repulsion_regulator(distance)
                shepherd_repulsion = shepherd_repulsion + M_e*force
            
        return self.normalize(shepherd_repulsion)   


    def compute_sheep_attraction(self, SheepIndex, sheepPosition):
        # Determine number of sheep within the sheep-sheep attraction range
        neighbourhood_idx = np.zeros(self.NumberOfSheep).astype(bool)
        for j in range(self.NumberOfSheep):        
            if (j != SheepIndex):
                distance = self.dist_p2p(sheepPosition, self.SheepMatrix[j,0:2])
                if distance <= self.NeighbourhoodRange:
                    neighbourhood_idx[j] = True
        
        neighbourSheepPositions = self.SheepMatrix[neighbourhood_idx,0:2]
        LCM = np.mean(neighbourSheepPositions, axis=0)
        sheep_attraction = LCM - sheepPosition
        return self.normalize(sheep_attraction)   


    def compute_obstacle_avoidance(self, sheepPosition):
        # Determine number of obstacles within the avoidance range
        avoidance_force = np.array([0,0])
        for i in range(len(self.Obstacles)):
            distance = self.dist_p2p(sheepPosition, self.Obstacles[i,0:2]) - self.Obstacles[i,2]*np.sqrt(2) # for square obstacle
            if distance <= self.AvoidanceRadius:
                v_sheep2obstacle = [self.Obstacles[i,0] - sheepPosition[0], self.Obstacles[i,1] - sheepPosition[1]]
                nv1, nv2 = self.perpendicular_vector_pair(v_sheep2obstacle)
                angle1 = self.angle_vectors(v_sheep2obstacle, nv1)
                angle2 = self.angle_vectors(v_sheep2obstacle, nv2)
                M_a = self.compute_obstacle_avoidance_regulator(distance)
                if angle1 < angle2:
                    avoidance_force = avoidance_force + M_a * np.array(nv1)
                else:
                    avoidance_force = avoidance_force + M_a * np.array(nv2)
        return self.normalize(avoidance_force)
        
    
    def compute_forces(self, SheepIndex, sheepPosition):
        ShepherdRepulsionDirection = self.compute_shepherd_repulsion(sheepPosition)
        if self.v_len(ShepherdRepulsionDirection) == 0:
            PreviousDirection = np.array([0,0])
        else:
            PreviousDirection = self.compute_previous_direction(SheepIndex)
            
        AttractionDirection = self.compute_sheep_attraction(SheepIndex, sheepPosition)

        RepulsionSheepDirection = self.compute_sheep_repulsion(SheepIndex, sheepPosition)
        
        if self.Obstacles is None:
            AvoidanceForce = np.array([0,0])
        else:
            AvoidanceForce = self.compute_obstacle_avoidance(sheepPosition)
        
        return (PreviousDirection, RepulsionSheepDirection, ShepherdRepulsionDirection, AttractionDirection, AvoidanceForce)
    

    def obstacle_collision(self, sheepPosition):
        collision = False
        for i in range(len(self.Obstacles)):
            if self.dist_p2p(sheepPosition, self.Obstacles[i,0:2]) <= self.Obstacles[i,2]*np.sqrt(2):
                collision = True
                break
            
        return collision

    
    def update(self, SheepMatrix, ShepherdMatrix):
        self.SheepMatrix = SheepMatrix
        self.ShepherdMatrix = ShepherdMatrix
        UpdatedSheepMatrix = np.zeros([self.NumberOfSheep, 5])
        Collision = False
        for i in range(self.NumberOfSheep):
            sheepPosition = self.SheepMatrix[i,0:2]
            PreviousDirection, RepulsionSheepDirection, ShepherdRepulsionDirection, AttractionDirection, AvoidanceForce = self.compute_forces(i, sheepPosition)
            # Find the total force
            CumulativeForce = self.WeightOfInertia*PreviousDirection \
                            + self.WeightRepellFromOtherSheep*RepulsionSheepDirection \
                            + self.WeightRepellFromShepherd*ShepherdRepulsionDirection \
                            + self.WeightAttractionToLCM*AttractionDirection \
                            + self.AvoidanceWeight*AvoidanceForce \
                            + self.NoiseLevel*np.clip(np.random.randn(2),-1,1)                
            
            # Find the normalised total force
            NormalisedForce = self.normalize(CumulativeForce)
            # Compute the next position of the shepherd
            NewPosition = sheepPosition + NormalisedForce * self.SheepStep
            
            ## If next position collide with obstacles, then do not move:
            if self.Obstacles is not None:
                if self.obstacle_collision(NewPosition):
                    UpdatedSheepMatrix[i,:] = self.SheepMatrix[i,:]
                    Collision = True
                else:
                    UpdatedSheepMatrix[i,0:2] = NewPosition
            else:
                UpdatedSheepMatrix[i,0:2] = NewPosition
            # New directional angle
            UpdatedSheepMatrix[i,2] = np.arctan2(NormalisedForce[1],NormalisedForce[0])
            UpdatedSheepMatrix[i,4] = self.SheepMatrix[i,4]
            ## Bound the movement within the paddock
            if (UpdatedSheepMatrix[i,0] < 0 ):
                UpdatedSheepMatrix[i,0] = 0
                     
            if (UpdatedSheepMatrix[i,1] < 0 ):
                UpdatedSheepMatrix[i,1] = 0
                             
            if( UpdatedSheepMatrix[i,0] > self.PaddockLength ):
                UpdatedSheepMatrix[i,0] = self.PaddockLength
                             
            if (UpdatedSheepMatrix[i,1] > self.PaddockLength ):
                UpdatedSheepMatrix[i,1] = self.PaddockLength
            
        return (UpdatedSheepMatrix, Collision)