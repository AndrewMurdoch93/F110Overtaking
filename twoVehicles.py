from argparse import Namespace
import math
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from pyglet.gl import GL_POINTS
import pandas as pd
import yaml
import sys


import cubic_spline_planner
import drivingAlgorithms
import functions
from f1tenth_gym.f110_env import F110Env
import mapping
import pathTracker
import rewardSignal
import trackCenterline

from drivingAlgorithms import purePursuitLineFollower
from drivingAlgorithms import LQRLineFollower
from drivingAlgorithms import MPCLineFollower



def race(scenarioFilename, numberEpisodes, numberRuns, saveFilepath, render=True):
    """
    Function called to test a single agent
    """

    agentTrainer = AgentTrainer(scenarioFilename, render)
    agentTrainer.executeRuns(numberEpisodes, numberRuns, saveFilepath, vehicleModelParamList=None)

 

class AgentTrainer():
    """
    This class manages the training of agents
    """
    
    def __init__(self, scenarioFilename, render):
        
        # Get scenario configuration
        self.scenarioParams = functions.openConfigFile('scenarios/'+scenarioFilename)
        self.render=render

        # Initialise the environment and track centerline
        self.numVehicles = len(self.scenarioParams.drivingAlgorithmConfigs)
        self.env = F110Env(map = self.scenarioParams.map, num_agents = self.numVehicles)
        obs, _, _, _ = self.env.reset(np.zeros((self.numVehicles, 4)))
        self.trackLine = trackCenterline.TrackLine(map = self.scenarioParams.map, numVehicles=self.numVehicles)
        self.trackLine.reset(obs)


        # Get configuration of driving algorithms
        self.drivingAlgorithmConfigs = []
        for drivingAlgorithmConfig in self.scenarioParams.drivingAlgorithmConfigs:
            self.drivingAlgorithmConfigs.append(functions.openConfigFile('drivingAlgorithms/'+drivingAlgorithmConfig))
        
        
        self.vehicleConfigs = []
        for vehicleConfig in self.scenarioParams.vehicleConfigs:
            self.vehicleConfigs.append(functions.openConfigFile('vehicleModel/'+vehicleConfig))
        


        # Initialise driving algorithms
        self.drivers = self.initialiseDrivingAlgorithms()

        # self.rewardSignal = rewardSignal.RewardSignal(conf)
        # self.learnSteps = 10


        #Add render callbacks
        def renderCallback(renderObject):
            self.trackLine.renderCenterline(renderObject)

            # if self.conf.drivingAlgorithm == "partial end-to-end":
            #     self.drivingAlgorithm.steeringControl.renderPurePursuit(renderObject)

        # Must happen after selecting driving algorithm due to placement of render callback functions 
        self.env.add_render_callback(renderCallback)

        self.slidingWindow = 10
        
        self.resultsDict = {
            'run': [],
            'episode': [],
            'episodeReward': [],
            'episodeProgress': [],
            'episodeLapTime': [],
            'episodeCrash': []
        }

        self.resultsDataframe = pd.DataFrame(self.resultsDict)
    

    def initialiseDrivingAlgorithms(self):
        """
        Selects and initialises a list of driving algorithms
        """
                
        drivers = []
        for idx, (drivingAlgorithmConfig, vehicleConfig) in enumerate(zip(self.drivingAlgorithmConfigs, self.vehicleConfigs)):
            if drivingAlgorithmConfig.drivingAlgorithm == "purePursuitLineFollower":
                if drivingAlgorithmConfig.globalPlan == "trackCenterLine":
                    drivers.append(purePursuitLineFollower.purePursuitLineFollower(conf=drivingAlgorithmConfig, line=self.trackLine, vehicleNumber=idx))
            
            if drivingAlgorithmConfig.drivingAlgorithm == "LQRLineFollower":
                if drivingAlgorithmConfig.globalPlan == "trackCenterLine":
                    drivers.append(LQRLineFollower.LQRLineFollower(conf=drivingAlgorithmConfig, line=self.trackLine, vehicleNumber=idx))

            if drivingAlgorithmConfig.drivingAlgorithm == "MPCLineFollower":
                if drivingAlgorithmConfig.globalPlan == "trackCenterLine":
                    drivers.append(MPCLineFollower.MPCLineFollower(algorithmConf=drivingAlgorithmConfig, vehicleConf=vehicleConfig, line=self.trackLine, vehicleNumber=idx))



        return drivers


    def generateInitialPoses(self):
        
        """
        Generates an initial pose for the ego vehicle by selecting a random point on the centerline
        Target vehicles spawn ahead of the ego vehicle
        """

        spawnIdxs = np.zeros(self.numVehicles)
        initialPoses = np.zeros((self.numVehicles, 4))

        spawnIdxs[0] = np.random.randint(low = 0, high = self.trackLine.numberIdxs)
        for i in range(1, self.numVehicles):
            spawnIdxs[i] = spawnIdxs[i-1]+15 # Cars spawn in front of each other

            # Make sure that generated trackline indeces are valid
            if spawnIdxs[i] > self.trackLine.numberIdxs:
                spawnIdxs[i] %= self.trackLine.numberIdxs
            if  spawnIdxs[i] < 0:
                spawnIdxs[i] += self.trackLine.numberIdxs
        
        for i in range(self.numVehicles):
            xStart = self.trackLine.cx[int(spawnIdxs[i])]
            yStart = self.trackLine.cy[int(spawnIdxs[i])]
            yawStart = self.trackLine.cyaw[int(spawnIdxs[i])]
            velStart = np.random.uniform(low=3, high=4)
            initialPoses[i, :] = np.array([xStart, yStart, yawStart, velStart]) 

        return initialPoses



    def executeRuns(self, numberEpisodes, numberRuns, saveFilepath, vehicleModelParamList=None):
        
        for run in range(numberRuns):
            print('Run: ', str(run))
            
            
            for idx, driver in enumerate(self.drivers):
                driver.reset(run=run)

            self.executeEpisodes(numberEpisodes, saveFilepath, vehicleModelParamList)


    def executeEpisodes(self, numberEpisodes, saveFilepath, vehicleModelParamList=None, run=0):


        # filePath = self.scenarioParams.savePath + '.csv'


        for episode in range(numberEpisodes):
            
            episodeCrash, episodeReward, episodeProgress, lapTime = self.executeOneEpisode()

            # Update information for training run
            dataRow = {
            'run': run,
            'episode': episode,
            'episodeReward': episodeReward,
            'episodeProgress': episodeProgress,
            'episodeLapTime': lapTime,
            'episodeCrash': episodeCrash
            }

            # self.resultsDataframe = pd.concat([self.resultsDataframe, pd.DataFrame([dataRow])])
            
            # # Update sliding averages over multiple episodes
            # averageReward = np.mean(episodesReward[-self.slidingWindow:])
            # averageProgress = np.mean(episodesProgress[-self.slidingWindow:])

            # print(f"{'Run':4s} {run:2.0f} {'| Episode':8s} {episode:5.0f} {'| Reward: ':8s} {episodeReward:6.2f} {'| Progress: ':12s} {episodeProgress/self.trackLine.distance[-1]*100:7.2f} {'%':1s}  {'| Lap Time: ':12s} {lapTime:3.2f}  ")

            # print('Episode: ', str(episode), ' | ', dataRow)

            # if ((episode%10==0) or (episode==numberEpisodes-1)) and (episode!=0):
            #     if training==True:
            #         print("Agent was saved: ", self.conf.name)
            #         self.drivingAlgorithm.agent.save_agent(name=self.conf.name, run=run)
                
                # self.resultsDataframe.to_csv(filePath)
                # print("Data was saved: ", self.conf.name)

    
    

    def executeOneEpisode(self, initialPoses=None, saveTrajectory=False, vehicleModelParamList=None):
        
        """
        Inputs:
        initialPoses: Specified as np.array([[xStart, yStart, yawStart], [xStart, yStart, yawStart], ... ]). To generate random starting points, specify empty array np.array([[]])
        saveTrajectory: True/False. If True, function returns trajectory as additional output.
        training: True/Flase.
        render: True/False.
        vehicleModelParamDict: Dictionary with vehicle model parameters. If empty, default parameters are used. 
        """


        # Get initial pose of vehicle, if not specified
        if initialPoses is None:
            initialPoses =  self.generateInitialPoses()
        
        # Reset the vehicle model, if necessary
        if vehicleModelParamList is not None:
            for i in range(self.numVehicles):
                self.env.update_params(vehicleModelParamList[i], index=0)
        
        # Reset componenets of the environmnet (vehicle and trackline)
        obs, step_reward, done, info = self.env.reset(initialPoses)
        next_obs = obs
        self.trackLine.reset(obs)
        
        # Reset variables to store episode data
        lapTime = 0.
        episodeProgress = np.zeros(self.numVehicles)
        episodeCrash = False
        episodeTrajectory = []
        episodeProgresses = []

        # Reset episode time to 0
        timeStep=0

        # Define shape of control action
        controlActions = np.zeros((self.numVehicles, 2))

        # Complete one episode
        while not done and np.any((episodeProgress <= 5*self.trackLine.distance[-1])):
            
            # Get actions from each algorithm
            for idx, driver in enumerate(self.drivers):
                controlActions[idx,:] = driver.stepDrivingAlgorithm(next_obs)

            # Simulation step
            next_obs, step_time, done, info = self.env.step(controlActions)
            
            # Get progress along centerline
            timeStepProgress = self.trackLine.findTimeStepProgresses(next_obs)
            episodeProgress += timeStepProgress

            obs = next_obs
            
            timeStep+=1
            
            if saveTrajectory==True:
                episodeTrajectory.append([obs])
                episodeProgresses.append(episodeProgress)


            if self.render==True:
                self.env.render(mode='human_fast')
        
        if done==True:
            episodeCrash=1
            lapTime=np.nan
            print("Episode completed with crash")

        if done==False:
            episodeCrash=0
            lapTime=timeStep/100
            print("Episode completed")



        # if saveTrajectory==False:
        #     return episodeCrash, episodeReward, episodeProgress, lapTime
        # elif saveTrajectory==True:
        #     return episodeCrash, episodeReward, episodeProgress, lapTime, episodeTrajectory, episodeProgresses
        


# race(scenarioFilename='twoLineFollowers', numberEpisodes=10, numberRuns=1, saveFilepath='experimentsData/testingData/twoLineFollowers', render=True) 
# race(scenarioFilename='LQRLineFollower', numberEpisodes=10, numberRuns=1, saveFilepath='experimentsData/testingData/LQRLineFollower', render=True) 
race(scenarioFilename='MPCLineFollower', numberEpisodes=1, numberRuns=1, saveFilepath='experimentsData/testingData/MPCFollower', render=True) 





