from argparse import Namespace
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pyglet.gl import GL_POINTS
import pandas as pd
import yaml
import sys


import agentTD3
import cubic_spline_planner
import functions
from F1TenthRacingDRL.f1tenth_gym import F110Env
import mapping
import pathTracker



def  selectDrivingAlgorithm(conf):
    if conf.drivingAlgorithm == "purePursuit":
        drivingAlgorithm = pathTracker.purePursuit(conf)
        drivingAlgorithm.read_centerline_waypoints_csv()
    elif  conf.drivingAlgorithm == "end-to-end":
        drivingAlgorithm = agentTD3.agent(conf)
    else:
        raise ValueError(f"Planner type not recognised: {conf.drivingAlgorithm}")    
    
    return drivingAlgorithm


def openConfigFile(configFileName):
    with open(configFileName+'.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    return conf




def testDrivingAlgorithm(configFileName):


    def renderCallback(renderObject):
        if conf.drivingAlgorithm=='purePursuit':
            drivingAlgorithm.renderPurePursuit(renderObject)


    conf = openConfigFile(configFileName)

    env = F110Env(map=conf.map_path, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[0., 0., 0.]])) 


    drivingAlgorithm = selectDrivingAlgorithm(conf)


    # Must happen after selecting driving algorithm due to placement of render callback functions 
    if conf.render==True:
        env.add_render_callback(renderCallback)
        env.render()

    # Define shape of action
    actions = np.zeros((1,2)) 



    for episode in range(conf.testEpisodes):
        
        obs, step_reward, done, info = env.reset(np.array([[0., 0., 0.]]))
        drivingAlgorithm = selectDrivingAlgorithm(conf)

        # Define useful variables to store data
        lapTime = 0.
        obsHistory = []
        
        # Complete one episode
        while not done and (lapTime <= conf.testEpisodeTimeLimit):

            # Select an action (1 vehicle)
            actions[0] = drivingAlgorithm.selectAction(obs)

            # Stepping through the environment
            obs, step_time, done, info = env.step(actions)
            
            if conf.render==True:
                env.render(mode='human_fast')
            
            obsHistory.append(obs)
            lapTime += step_time


        print('Episode is finished')
        print(lapTime)
        # np.any(env.collisions), then crashed

        xHistory = np.zeros((len(obsHistory)))
        yHistory = np.zeros((len(obsHistory)))

        for i, obs in enumerate(obsHistory):
            xHistory[i] = obs['poses_x'][0]
            yHistory[i] = obs['poses_y'][0]





def trainAgent(configFileName):
    
    """
    Function called to train a single agent
    """

    conf = openConfigFile(configFileName)
    agentTrainer = AgentTrainer(conf)
    agentTrainer.trainAgent()




class AgentTrainer():
    
    """
    This class manages the training of agents
    """
    
    def __init__(self, conf):
        
        def renderCallback(renderObject):
            self.trackLine.renderCenterline(renderObject)

            if conf.drivingAlgorithm == "partial end-to-end":
                self.architecture.steeringControl.renderPurePursuit(renderObject)


        self.conf = conf
        self.numAgents = 1
        self.env = F110Env(map=conf.map_path, num_agents=self.numAgents)
        obs, _, _, _ = self.env.reset(np.array([[0., 0., 0.]])) 

        self.rewardSignal = RewardSignal(conf)
        self.trackLine = TrackLine(conf)
        self.trackLine.reset(obs)

        self.agent = agentTD3.agent(conf)

        
        if conf.drivingAlgorithm == 'end-to-end':
            self.architecture = EndtoEndArchitecture(conf)
        if conf.drivingAlgorithm == 'partial end-to-end':
            self.architecture = PartialEndtoEndArchitecture(conf, self.trackLine)
        

        # Must happen after selecting driving algorithm due to placement of render callback functions 
        if conf.render==True:
            self.env.add_render_callback(renderCallback)


    def generateInitialPose(self):
        
        spawnIdx = np.random.randint(low = 0, high = self.trackLine.numberIdxs)
        xStart = self.trackLine.cx[spawnIdx]
        yStart = self.trackLine.cy[spawnIdx]
        yawStart = self.trackLine.cyaw[spawnIdx]
        
        return xStart, yStart, yawStart


    def trainAgent(self):
            
        # Define shape of action
        nn_actions = np.zeros((self.numAgents, 2)) 
        
        # Variables containing information for all episodes (also used to calculate sliding average)
        slidingWindow = 10
        episodesReward = []
        episodesProgress = []
        episodesLapTime = []
        episodesCrash = []
        runs = []

        run=1

        for episode in range(self.conf.trainEpisodes):
            
            # Reset componenets of the environmnet (vehicle and trackline)
            xStart, yStart, yawStart =  self.generateInitialPose()
            obs, step_reward, done, info = self.env.reset(np.array([[xStart, yStart, yawStart]]))
            self.trackLine.reset(obs)
            
            # Reset variables to store episode data
            lapTime = 0.
            episodeProgress = 0.
            episodeReward = 0.
            episodeCrash = False

            # Complete one episode
            while not done and (lapTime <= self.conf.trainEpisodeTimeLimit) and episodeProgress<=self.trackLine.distance[-1]:

                # Select an action (1 vehicle)
                nn_actions[0] = self.agent.choose_action(self.architecture.transformObservation(obs))
                
                if self.conf.drivingAlgorithm=='end-to-end':
                    #Transform action to steering, velocity
                    actions =  np.array([self.architecture.transformAction(nn_actions[0])])
                
                if self.conf.drivingAlgorithm=='partial end-to-end':
                    self.architecture.generateLocalPath(obs, nn_actions[0])
                
                # Stepping through the environment (multiple times, due to multiple control steps)
                for _ in range(self.conf.controlSteps):
                    
                    if self.conf.drivingAlgorithm=='partial end-to-end':
                        actions = np.array([self.architecture.generateControlAction(nn_actions[0], obs)])

                    next_obs, step_time, done, info = self.env.step(actions)
                    if done:
                        episodeCrash=True
                        break
                
                # Get information for 1 time step 
                timeStepProgress = self.trackLine.findTimeStepProgress(next_obs)
                reward = self.rewardSignal.calculateReward(timeStepProgress, done)
                lapTime += step_time*self.conf.controlSteps
                if episodeCrash==True:
                    lapTime=np.nan


                # Update information for entire episode
                episodeReward += reward
                episodeProgress += timeStepProgress          
                
                # Transform observation for compatibility with DNN
                nn_obs = self.architecture.transformObservation(obs)
                nn_next_obs = self.architecture.transformObservation(next_obs)
                
                self.agent.store_transition(nn_obs, nn_actions[0], reward, nn_next_obs, int(done))
                self.agent.learn()
                
                obs = next_obs

                if self.conf.render==True:
                    self.env.render(mode='human_fast')
            
            # Update information for training run
            episodesReward.append(episodeReward)
            episodesProgress.append(episodeProgress)
            episodesLapTime.append(lapTime)
            episodesCrash.append(episodeCrash)
            runs.append(run)
            
            # Update sliding averages over multiple episodes
            averageReward = np.mean(episodesReward[-slidingWindow:])
            averageProgress = np.mean(episodesProgress[-slidingWindow:])

            print(f"{'Episode':8s} {episode:5.0f} {'| Reward':8s} {episodeReward:6.2f} {'| Progress':12s} {episodeProgress:3.2f} {'| Average Reward':15s} {averageReward:3.2f} {'| Average Progress':18s} {averageProgress:3.2f} ")

            if (episode%10==0) and (episode!=0):
                self.agent.save_agent(name=self.conf.name, run=run)
                self.saveTrainingRunData(episodesReward, episodesProgress, episodesLapTime, episodesCrash, runs)
                print("Agent and training data were saved")


        
    def saveTrainingRunData(self, episodesReward, episodesProgress, episodeslapTime, episodesCrash, runs):
        
        directory = 'trainingData/'
        fileName = self.conf.name
        filePath = directory + '/' + fileName

        trainingData = {
            'run': runs,
            'Crash': episodesCrash,
            'Reward': episodesReward,
            'Progress': episodesProgress,
            'Lap time': episodeslapTime
        }

        if not os.path.exists(directory):
            os.mkdir(directory) 

        trainingDataframe = pd.DataFrame(trainingData)
        trainingDataframe.to_csv(filePath)

    




class PartialEndtoEndArchitecture():
    
    """
    This class handles the interface between the agent and environment
    """
    
    def __init__(self, conf, trackLine):
        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity
        self.LiDARRange = 30
        
        #still need to automate this
        self.maxX = 6.8
        self.maxY = 4.9
        self.maxTheta = 2*np.pi 

        # Define components
        self.trackLine = trackLine
        self.steeringControl = pathTracker.purePursuit(conf)


    def generateLocalPath(self, obs, nn_action):
        
        track_width = 1
        ds=0.1

        s_0_ind, s_0, n_0, theta = self.trackLine.convert_xy_obs_to_sn(obs)

        # plt.plot(self.trackLine.cx, self.trackLine.cy)
        # plt.plot(obs['poses_x'][0], obs['poses_y'][0], 'x')
        # plt.plot(self.trackLine.cx[s_0_ind], self.trackLine.cy[s_0_ind], 'o')
        # plt.show()

        s_1 = s_0 + 3
        s_2 = s_1 + 2
        n_1 = nn_action[0]*track_width/2
        # theta = functions.sub_angles_complex(obs['poses_theta'][0], self.trackLine.cyaw[s_0_ind])
        A = np.array([[3*s_1**2, 2*s_1, 1, 0], [3*s_0**2, 2*s_0, 1, 0], [s_0**3, s_0**2, s_0, 1], [s_1**3, s_1**2, s_1, 1]])
        B = np.array([0, theta, n_0, n_1])
        x = np.linalg.solve(A, B)
        s = np.linspace(s_0, s_1)
        n = x[0]*s**3 + x[1]*s**2 + x[2]*s + x[3]
        s = np.concatenate((s, np.linspace(s_1, s_2)))
        s = np.mod(s, self.trackLine.distance[-1])
        n = np.concatenate((n, np.ones(len(np.linspace(s_1, s_2)))*n_1))



        pathX, pathY, pathYaw, pathDs = self.trackLine.convert_sn_path_to_xy(s, n)

        self.steeringControl.record_waypoints(cx = pathX, cy = pathY, cyaw = pathYaw)

        
        # plt.plot(obs['poses_x'], obs['poses_y'], 'x')
        # plt.plot(self.trackLine.cx, self.trackLine.cy)
        # plt.plot(pathX, pathY)
        # plt.show()



    
    def generateControlAction(self, nn_action, obs):

        """
        Transforms an action selected by the neural network (i.e., scaled to the range (-1,1))
        to steering and velocity commands, that can be interpreted by the simulator
        
        *NB: Transforms a single action.
        """

        # Find target lookahead point on centerline
        target_index, _ = self.steeringControl.search_target_waypoint(obs['poses_x'][0], obs['poses_y'][0], obs['linear_vels_x'][0])
        
        # Get desired steering angle based on target point
        delta_ref, target_index = self.steeringControl.pure_pursuit_steer_control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], target_index)

        # Select velocity
        velocity_ref = nn_action[1]*(self.maxVelocity-self.minVelocity)/2 + self.minVelocity + (self.maxVelocity-self.minVelocity)/2

        # Combine steering and velocity into an action
        action = np.array([delta_ref, velocity_ref])

        return action

    def transformObservation(self, obs):

        pose = np.array([obs['poses_x'][0]/(self.maxX), obs['poses_y'][0]/(self.maxY), obs['poses_theta'][0]/(self.maxTheta),  obs['linear_vels_x'][0]/(self.maxVelocity)])
        nn_observation = np.concatenate((pose, obs['scans'][0]/30.0))


        return nn_observation

    
    



class EndtoEndArchitecture():
    
    """
    This class handles the interface between the agent and environment
    """
    
    def __init__(self, conf):
        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity
        self.LiDARRange = 30
        
        #still need to automate this
        self.maxX = 6.8
        self.maxY = 4.9
        self.maxTheta = 2*np.pi 
 
    
    def transformAction(self, nn_action):

        """
        Transforms an action selected by the neural network (i.e., scaled to the range (-1,1))
        to steering and velocity commands, that can be interpreted by the simulator
        
        *NB: Transforms a single action.
        """

        delta_ref = nn_action[0]*0.4
        velocity_ref = nn_action[1]*(self.maxVelocity-self.minVelocity)/2 + self.minVelocity + (self.maxVelocity-self.minVelocity)/2
        
        action = np.array([delta_ref, velocity_ref])

        return action

    def transformObservation(self, obs):

        pose = np.array([obs['poses_x'][0]/(self.maxX), obs['poses_y'][0]/(self.maxY), obs['poses_theta'][0]/(self.maxTheta),  obs['linear_vels_x'][0]/(self.maxVelocity)])
        nn_observation = np.concatenate((pose, obs['scans'][0]/30.0))


        return nn_observation





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

    




class RewardSignal():

    """
    A class that manages the reward signal
    Reward signal is dependent on progress along centerline
    timeStep progress is calculated in main while loop
    """

    def __init__(self, conf):

        self.distanceReward = conf.distanceReward
        self.timeStepPenalty = conf.timeStepPenalty
        self.collisionPenalty = conf.collisionPenalty
        

    def calculateReward(self, timeStepProgress, done):
        
        # timeStepProgress = self.findTimeStepProgress(obs)*0.1


        if not done:
            reward = self.timeStepPenalty +  timeStepProgress * self.distanceReward
        if done:
            reward = self.collisionPenalty

        return reward



        











if __name__=='__main__':

    # testDrivingAlgorithm('config_example_map')
    trainAgent('configTD3Agent')