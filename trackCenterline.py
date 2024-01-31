import math
import numpy as np
import pandas as pd
from pyglet.gl import GL_POINTS
import sys

import functions



class TrackLine():
    
    """
    This class stores and manages the centerline
    Tracks the progress of the vehicle along the centerline
    Visualises the centerline
    """

    def __init__(self, conf):
        self.mapPath = conf.map_path
        self.cx, self.cy, self.cyaw, self.ccurve, self.distance, self.csp = self.loadCenterline()
        self.numberIdxs = len(self.cx)


    def loadCenterline(self):
        centerlineDataframe = pd.read_csv(sys.path[0] + '/maps/' + self.mapPath + '_centerline.csv')
        cx = np.array(centerlineDataframe['x'])
        cy = np.array(centerlineDataframe['y'])
        cyaw = np.array(centerlineDataframe['yaw'])
        distance = np.array(centerlineDataframe['distance'])

        cx, cy, cyaw, ccurve, distance, csp = functions.generate_line(cx, cy)

        return cx, cy, cyaw, ccurve, distance, csp

    
    def reset(self, obs):
        self.initialCenterlinePoint = self.findClosestCenterlinePoint(obs)
        self.oldClosestCenterlinePointIdx = self.initialCenterlinePoint
    

    def findTimeStepProgress(self, obs):

        newClosestCenterlinePointIdx = self.findClosestCenterlinePoint(obs)
        centerlineIdxDiff = newClosestCenterlinePointIdx - self.oldClosestCenterlinePointIdx

        # Vehilce is stationary or travelling perpendicular to centerline
        if  centerlineIdxDiff==0:
            timeStepProgress = 0
        
        # Vehicle is travelling forwards along centerline
        elif (0 < centerlineIdxDiff < int(self.numberIdxs/2)):   
            timeStepProgress = newClosestCenterlinePointIdx - self.oldClosestCenterlinePointIdx
        
        # Vehicle is crossing start location, going forward
        elif centerlineIdxDiff < -int(self.numberIdxs/2):   
            timeStepProgress = (self.numberIdxs-np.abs(self.oldClosestCenterlinePointIdx)) + newClosestCenterlinePointIdx

        # Vehicle is travelling backwards
        elif -self.numberIdxs < centerlineIdxDiff < 0:  
            timeStepProgress = newClosestCenterlinePointIdx - self.oldClosestCenterlinePointIdx

        # Vehicle is crossing start location going backwards
        elif centerlineIdxDiff>=int(self.numberIdxs/2):    
            timeStepProgress = -(self.oldClosestCenterlinePointIdx+(np.abs(self.numberIdxs - newClosestCenterlinePointIdx)))

        self.oldClosestCenterlinePointIdx =  newClosestCenterlinePointIdx

        # Convert from idx to distance
        timeStepDistanceProgress = timeStepProgress*0.1

        return timeStepDistanceProgress


    def findClosestCenterlinePoint(self, obs):
        
        dx = [obs['poses_x'][0] - irx for irx in self.cx]
        dy = [obs['poses_y'][0] - iry for iry in self.cy]
        distances = np.hypot(dx, dy)    
        ind = np.argmin(distances)
        
        return ind

    def findClosestDistanceToCenterline(self, obs):
        
        dx = [obs['poses_x'][0] - irx for irx in self.cx]
        dy = [obs['poses_y'][0] - iry for iry in self.cy]
        distances = np.hypot(dx, dy)    
        
        idx = np.argmin(distances)
        closestDist = distances[idx]

        return closestDist


    def convert_xy_obs_to_sn(self, obs):
        
        """
        Converts the vehicle pose to Frenet frame coordinates
        """

        ds = self.distance[1]

        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        yaw = obs['poses_theta'][0]

        ind = self.findClosestCenterlinePoint(obs)
        
        s = ind*ds                              # Exact position of s
        n = self.findClosestDistanceToCenterline(obs)   # n distance (unsigned), not interpolated

        # Get sign of n by comparing angle between (x,y) and (s,0), and the angle of the centerline at s
        xy_angle = np.arctan2((y-self.cy[ind]),(x-self.cx[ind]))      # angle between (x,y) and (s,0)
        yaw_angle = self.cyaw[ind]                               # angle at s
        angle = functions.sub_angles_complex(xy_angle, yaw_angle)     
        if angle >=0:       # Vehicle is above s line
            direction=1     # Positive n direction
        else:               # Vehicle is below s line
            direction=-1    # Negative n direction

        n = n*direction   # Include sign in n 

        theta = functions.sub_angles_complex(yaw, yaw_angle)

        return ind, s, n, theta

    
    def convert_sn_path_to_xy(self, s, n):
        
        """
        Converts a trajectory of Frenet frame (s,n) points to Cartesian (x,y) coordinates
        """
        
        x = []
        y = []
        yaw = []
        ds = []

        for i in range(len(s)):
            ix, iy = self.csp.calc_position(s[i])
            if ix is None:
                break
            i_yaw = self.csp.calc_yaw(s[i])
            ni = n[i]
            fx = ix + ni * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + ni * math.sin(i_yaw + math.pi / 2.0)
            x.append(fx)
            y.append(fy)

        # calc yaw and ds
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            yaw.append(math.atan2(dy, dx))
            ds.append(math.hypot(dx, dy))
            yaw.append(yaw[-1])
            ds.append(ds[-1])

        return x, y, yaw, ds



    def renderCenterline(self, renderObject):
        
        #Render centerline points

        points = np.vstack((self.cx,self.cy)).T
        scaled_points = 50.*points
        for i in range(points.shape[0]):
            b = renderObject.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [250, 250, 250]))

    

