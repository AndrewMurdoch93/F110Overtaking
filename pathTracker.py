import math
import numpy as np
import pandas as pd
import sys
import pyglet
from pyglet.gl import GL_POINTS

import functions


class purePursuit():
    
    def __init__(self, conf):

        # Parameters for initialisation
        self.conf = conf
        self.wheelbase = conf.wheelbase
        self.k = conf.k
        self.Lfc = conf.Lfc

        self.old_nearest_point_index = None
        
        # For visualisation
        self.canvas = {}
        self.batch = pyglet.graphics.Batch()



    # def read_centerline_waypoints_csv(self):

    #     centerlineDataframe = pd.read_csv(sys.path[0] + '/maps/' + self.conf.map_path + '_centerline.csv')
        
    #     self.record_waypoints(cx=np.array(centerlineDataframe['x']),
    #                         cy=np.array(centerlineDataframe['y']),
    #                         cyaw=np.array(centerlineDataframe['yaw'])
    #                         )
    
    
    def record_waypoints(self, cx, cy, cyaw):
        #Initialise waypoints for planner
        self.cx=cx
        self.cy=cy
        self.cyaw = cyaw
        self.old_nearest_point_index = None

    def search_target_waypoint(self, x, y, v):
        
        #If there is no previous nearest point - at the start
        if self.old_nearest_point_index == None:
            #Get distances to every point
            dx = [x - icx for icx in self.cx]
            dy = [y - icy for icy in self.cy]
            d = np.hypot(dx, dy)    
            ind = np.argmin(d)      #Get nearest point
            self.old_nearest_point_index = ind  #Set previous nearest point to nearest point
        else:   #If there exists a previous nearest point - after the start
            #Search for closest waypoint after ind
            ind = self.old_nearest_point_index  
            #self.ind_history.append(ind)
        
            distance_this_index = functions.distance_between_points(self.cx[ind], x, self.cy[ind], y)   
            
            while True:
                if (ind+1)>=len(self.cx):
                    # break
                    ind=0
                
                distance_next_index = functions.distance_between_points(self.cx[ind + 1], x, self.cy[ind + 1], y)
                
                if distance_this_index < distance_next_index:
                    break

                ind = ind + 1 if (ind + 1) < len(self.cx) else ind  #Increment index - search for closest waypoint
                
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind  

        Lf = self.k * v + self.Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > functions.distance_between_points(self.cx[ind], x, self.cy[ind], y):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1 
        
        return ind, Lf

    def pure_pursuit_steer_control(self, x, y, theta, v, pind):
            
            ind, Lf = self.search_target_waypoint(x, y, v)

            if pind >= ind:
                ind = pind

            if ind < len(self.cx):
                tx = self.cx[ind]
                ty = self.cy[ind]
            else:  # toward goal
                tx = self.cx[-1]
                ty = self.cy[-1]
                ind = len(self.cx) - 1

            alpha = math.atan2(ty - y, tx - x) - theta
            delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha) / Lf, 1.0)

            return delta, ind
    
    def selectAction(self, obs):
        
        # Select an action based on current observation
        # Action includes velocity
        # For now, velocity is constant

        #Find target lookahead point on centerline
        target_index, _ = self.search_target_waypoint(obs['poses_x'][0], obs['poses_y'][0], obs['linear_vels_x'][0])
        
        #Get desired steering angle based on target point
        delta_ref, target_index = self.pure_pursuit_steer_control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], target_index)

        # Select velocity
        velocity_ref = 4

        # Combine 
        action = np.array([delta_ref, velocity_ref])

        return action
    
    def renderPurePursuit(self, renderObject):

        #Render line that pure pursuit follower is tracking
        points = np.vstack((self.cx,self.cy)).T
        scaled_points = 50.*points
        
        for i in range(points.shape[0]):
            vertex  = renderObject.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [100, 0, 0]))




