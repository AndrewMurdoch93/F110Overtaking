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





def race(configFileNames, render=True, episodes=100):
    """
    Function called to test a single agent
    """
    conf1 = functions.openConfigFile(configFileNames[0])
    conf2 = functions.openConfigFile(configFileNames[1])
    agentTrainer = AgentTrainer(conf1=conf1, conf2=conf2)
    # agentTrainer.executeRuns(numberEpisodes=episodes, numberRuns=conf.runs, render=render, training=False, saveDirectory='experimentsData/testingData', saveFilename=conf.name, vehicleModelParamDict={})
    agentTrainer.executeRuns(numberEpisodes=episodes, numberRuns=1, render=True, training=False, saveDirectory='experimentsData/testingData', saveFilename=conf1.name, vehicleModelParamDict={})
 

class AgentTrainer():
    """
    This class manages the training of agents
    """
    
    def __init__(self, conf1, conf2):
        
        self.conf1 = conf1
        self.conf2 = conf2

        self.numAgents = 2
        self.env = F110Env(map=conf1.map_path, num_agents=self.numAgents)
        obs, _, _, _ = self.env.reset(np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]])) 

        # self.rewardSignal = rewardSignal.RewardSignal(conf)
        self.trackLine = trackCenterline.TrackLine(conf1)
        self.trackLine.reset(obs)


        self.drivingAlgorithm1 = drivingAlgorithms.purePursuitCenterlineFollower(conf1, self.trackLine)
        self.drivingAlgorithm2 = drivingAlgorithms.purePursuitCarFollower(conf2, self.trackLine)
        self.learnSteps = self.conf1.controlSteps


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
        

    def generateInitialPoses(self):
        


        spawnIdx1 = np.random.randint(low = 0, high = self.trackLine.numberIdxs)
        spawnIdx2 = (spawnIdx1 - 40)
        if spawnIdx2 < 0:
            spawnIdx2+=self.trackLine.numberIdxs
        

        xStart1 = self.trackLine.cx[spawnIdx1]
        yStart1 = self.trackLine.cy[spawnIdx1]
        yawStart1 = self.trackLine.cyaw[spawnIdx1]
        velStart1 = np.random.uniform(low=self.conf1.minVelocity, high=self.conf1.maxVelocity)

        xStart2 = self.trackLine.cx[spawnIdx2]
        yStart2 = self.trackLine.cy[spawnIdx2]
        yawStart2 = self.trackLine.cyaw[spawnIdx2]
        velStart2 = velStart1

        initialPoses = np.array([[xStart1, yStart1, yawStart1, velStart1], [xStart2, yStart2, yawStart2, velStart2]])

        return initialPoses

    def executeRuns(self, numberEpisodes, numberRuns, render, training, saveDirectory, saveFilename, vehicleModelParamDict):
        
        for run in range(numberRuns):
            print('Run: ', str(run))
            self.drivingAlgorithm1.reset(run=run, training=training)
            self.drivingAlgorithm2.reset(run=run, training=training)
            self.executeEpisodes(numberEpisodes=numberEpisodes, run=run, render=render, training=training, saveDirectory=saveDirectory, saveFilename=saveFilename, vehicleModelParamDict=vehicleModelParamDict)



    def executeEpisodes(self, numberEpisodes, run, render, training, saveDirectory, saveFilename, vehicleModelParamDict):
        
        filePath = saveDirectory + '/' + saveFilename + '.csv'


        for episode in range(numberEpisodes):
            
            episodeCrash, episodeReward, episodeProgress, lapTime = self.executeOneEpisode(initialPose=np.array([]), saveTrajectory=False, training=training, render=render, vehicleModelParamDict=vehicleModelParamDict)

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

    
    

    def executeOneEpisode(self, initialPose, saveTrajectory, training, render, vehicleModelParamDict):
        
        """
        Inputs:
        initialPose: Specified as np.array([[xStart, yStart, yawStart]]). To generate random starting points, specify empty array np.array([[]])
        saveTrajectory: True/False. If True, function returns trajectory as additional output.
        training: True/Flase.
        render: True/False.
        vehicleModelParamDict: Dictionary with vehicle model parameters. If empty, default parameters are used. 
        """

        if len(vehicleModelParamDict)!=0: 
            self.env.update_params(vehicleModelParamDict, index=0)

        # Get initial pose of vehicle, if not specified
        if initialPose.size==0:
            initialPoses =  self.generateInitialPoses()

        
        # Reset componenets of the environmnet (vehicle and trackline)
        obs, step_reward, done, info = self.env.reset(initialPoses)
        next_obs = obs
        
        self.trackLine.reset(obs)
        
        # Reset variables to store episode data
        lapTime = 0.
        episodeProgress = 0.
        episodeReward = 0.
        episodeCrash = False
        
        episodeTrajectory = []
        episodeProgresses = []

        i=0

        # Complete one episode
        while not done and (episodeProgress <= 2*self.trackLine.distance[-1]):
            
            if i % self.conf1.planSteps == 0:
                plan1 = self.drivingAlgorithm1.generatePlan(next_obs)
                plan2 = self.drivingAlgorithm2.generatePlan(next_obs)

            if i % self.conf1.controlSteps == 0:
                controlActions1 = self.drivingAlgorithm1.generateControlAction(plan1, next_obs, 0)
                controlActions2 = self.drivingAlgorithm2.generateControlAction(plan2, next_obs, 1)

            next_obs, step_time, done, info = self.env.step(np.array([controlActions1, controlActions2]))
            
            if ((i+1) % self.learnSteps == 0 and i>1) or (done==True):
                
                # Get information for 1 [learning] time step 
                timeStepProgress = self.trackLine.findTimeStepProgress(next_obs)
                # reward = self.rewardSignal.calculateReward(timeStepProgress, done)


                # Update information for entire episode
                # episodeReward += reward
                episodeProgress += timeStepProgress
                
                # if training == True:
                #     self.drivingAlgorithm.storeTransition(obs, reward, next_obs, int(done))
                #     self.drivingAlgorithm.learn()
                
                obs = next_obs
            
            i+=1
            
            if saveTrajectory==True:
                episodeTrajectory.append([obs])
                episodeProgresses.append(episodeProgress)


            if render==True:
                self.env.render(mode='human_fast')
        
        if done==True:
            episodeCrash=1
            lapTime=np.nan
            print("Episode completed with crash")

        if done==False:
            episodeCrash=0
            lapTime=i/100
            print("Episode completed")



        if saveTrajectory==False:
            return episodeCrash, episodeReward, episodeProgress, lapTime
        elif saveTrajectory==True:
            return episodeCrash, episodeReward, episodeProgress, lapTime, episodeTrajectory, episodeProgresses
        


race(configFileNames=['purePursuitController', 'purePursuitCarFollower'], render=True, episodes=100) 



