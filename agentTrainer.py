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



def trainAgent(configFileName, render=False):
    """
    Function called to train a single agent
    """
    conf = functions.openConfigFile(configFileName)
    agentTrainer = AgentTrainer(conf=conf)
    agentTrainer.executeRuns(numberEpisodes=conf.trainEpisodes, numberRuns=conf.runs, render=render, training=True, saveDirectory='experimentsData/trainingData', saveFilename=conf.name, vehicleModelParamDict={})
    


def testAgent(configFileName, render=True, episodes=100):
    """
    Function called to test a single agent
    """
    conf = functions.openConfigFile(configFileName)
    agentTrainer = AgentTrainer(conf=conf)
    agentTrainer.executeRuns(numberEpisodes=episodes, numberRuns=conf.runs, render=render, training=False, saveDirectory='experimentsData/testingData', saveFilename=conf.name, vehicleModelParamDict={})




def modelMismatchExperiment(configFileName, vehicleModelParamDict,  vehicleModelChangeDict, saveFilename, render=False, episodes=100):
    """
    Function called to test a single agent under model-mismatch conditions
    """
    conf = functions.openConfigFile(configFileName)
    agentTrainer = AgentTrainer(conf=conf)
    agentTrainer.executeMismatchExperiment(numberEpisodes=episodes, numberRuns=conf.runs, render=render, saveDirectory='experimentsData/modelMismatchData', saveFilename=saveFilename, vehicleModelStandardDict=vehicleModelParamDict, vehicleModelChangeDict=vehicleModelChangeDict)



def trainEvaluate(configFileName, numberTrainEpisodes, numberEvalEpisodes, evalInterval, render=False):
    """
    Function called to test a single agent
    """
    conf = functions.openConfigFile(configFileName)
    agentTrainer = AgentTrainer(conf=conf)
    agentTrainer.trainEvaluate(numberTrainEpisodes=numberTrainEpisodes, numberEvalEpisodes=numberEvalEpisodes, evalInterval=evalInterval, render=render, saveFilename=conf.name, vehicleModelParamDict={})




class AgentTrainer():
    
    """
    This class manages the training of agents
    """
    
    def __init__(self, conf):
        
        self.conf = conf

        self.numAgents = 1
        self.env = F110Env(map=conf.map_path, num_agents=self.numAgents)
        obs, _, _, _ = self.env.reset(np.array([[0., 0., 0., 0.]])) 

        self.rewardSignal = rewardSignal.RewardSignal(conf)
        self.trackLine = trackCenterline.TrackLine(conf)
        self.trackLine.reset(obs)
    
        if conf.drivingAlgorithm == 'end-to-end':
            self.drivingAlgorithm = drivingAlgorithms.EndtoEnd(conf)
            self.learnSteps = self.conf.controlSteps
        if conf.drivingAlgorithm == 'partial end-to-end':
            self.drivingAlgorithm = drivingAlgorithms.PartialEndtoEnd(conf, self.trackLine)
            self.learnSteps = self.conf.planSteps
        
        #Add render callbacks
        def renderCallback(renderObject):
            self.trackLine.renderCenterline(renderObject)

            if self.conf.drivingAlgorithm == "partial end-to-end":
                self.drivingAlgorithm.steeringControl.renderPurePursuit(renderObject)

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
        

    def generateInitialPose(self):
        
        spawnIdx = np.random.randint(low = 0, high = self.trackLine.numberIdxs)
        xStart = self.trackLine.cx[spawnIdx]
        yStart = self.trackLine.cy[spawnIdx]
        yawStart = self.trackLine.cyaw[spawnIdx]
        velStart = np.random.uniform(low=self.conf.minVelocity, high=self.conf.maxVelocity)

        return xStart, yStart, yawStart, velStart


    def trainEvaluate(self, numberTrainEpisodes, numberEvalEpisodes, evalInterval, render, saveFilename, vehicleModelParamDict):

        trainFilePath = 'experimentsData/trainingData/' + saveFilename + '.csv'
        evalFilePath = 'experimentsData/evaluationData/' + saveFilename + '.csv'
        evalDataDict = {
            'trainEpisode': [],
            'evalEpisode': [],
            'episodeReward': [],
            'episodeProgress': [],
            'episodeLapTime': [],
            'episodeCrash': []
            }
        evalDataframe = pd.DataFrame(evalDataDict)

        self.drivingAlgorithm.reset(run=0, training=True)

        for episode in  range(numberTrainEpisodes):
            
            self.drivingAlgorithm.agent.training = True

            episodeCrash, episodeReward, episodeProgress, lapTime = self.executeOneEpisode(initialPose=np.array([]), saveTrajectory=False, training=True, render=render, vehicleModelParamDict=vehicleModelParamDict)

            
            # Update information for training run
            dataRow = {
            'run': 0,
            'episode': episode,
            'episodeReward': episodeReward,
            'episodeProgress': episodeProgress,
            'episodeLapTime': lapTime,
            'episodeCrash': episodeCrash
            }

            self.resultsDataframe = pd.concat([self.resultsDataframe, pd.DataFrame([dataRow])])
            
            # # Update sliding averages over multiple episodes
            # averageReward = np.mean(episodesReward[-self.slidingWindow:])
            # averageProgress = np.mean(episodesProgress[-self.slidingWindow:])

            print(f"{'| Episode':8s} {episode:5.0f} {'| Reward: ':8s} {episodeReward:6.2f} {'| Progress: ':12s} {episodeProgress/self.trackLine.distance[-1]*100:7.2f} {'%':1s}  {'| Lap Time: ':12s} {lapTime:3.2f}  ")

            # print('Episode: ', str(episode), ' | ', dataRow)

            if ((episode%10==0) or (episode==numberTrainEpisodes-1)) and (episode!=0):
            
                print("Agent was saved: ", self.conf.name)
                self.drivingAlgorithm.agent.save_agent(name=self.conf.name, run=0)
                
                self.resultsDataframe.to_csv(trainFilePath)
                print("Training data was saved: ", self.conf.name)

           
            if ((episode%evalInterval==0) or (episode==numberTrainEpisodes-1)) and (episode!=0):
                
                print("Agent is being evaluated: ", self.conf.name)
                self.drivingAlgorithm.agent.training = False
                for evalEpisode in  range(numberEvalEpisodes):
                    episodeCrash, episodeReward, episodeProgress, lapTime = self.executeOneEpisode(initialPose=np.array([]), saveTrajectory=False, training=False, render=render, vehicleModelParamDict=vehicleModelParamDict)

                    dataRow = {
                        'trainEpisode': episode,
                        'evalEpisode': evalEpisode,
                        'episodeReward': episodeReward,
                        'episodeProgress': episodeProgress,
                        'episodeLapTime': lapTime,
                        'episodeCrash': episodeCrash
                        }
                    evalDataframe = pd.concat([evalDataframe, pd.DataFrame([dataRow])])

                    print(f"{'Evaluation episode':18s} {evalEpisode:5.0f} {'| Reward: ':8s} {episodeReward:6.2f} {'| Progress: ':12s} {episodeProgress/self.trackLine.distance[-1]*100:7.2f} {'%':1s}  {'| Lap Time: ':12s} {lapTime:3.2f}  ")

                evalDataframe.to_csv(evalFilePath)
                print("Evaluation data was saved: ", self.conf.name)




    def executeMismatchExperiment(self, numberEpisodes, numberRuns, render, saveDirectory, saveFilename, vehicleModelStandardDict, vehicleModelChangeDict):
        
        saveFilenameData = saveFilename + 'Data'
        saveFilenameParams = saveFilename + 'Params'
        saveFilePathParams = saveDirectory + '/' + saveFilenameParams + '.csv'

        paramDataframe =  pd.DataFrame(columns=list(vehicleModelChangeDict.keys()))

        vehicleParamDict = vehicleModelStandardDict.copy()

        for i in range(len(list(vehicleModelChangeDict.values())[0])):
            
            paramDict = {}

            for key in vehicleModelChangeDict:
                vehicleParamDict[key] = vehicleModelChangeDict[key][i]
                print(key, ': ', str(vehicleModelChangeDict[key][i]))

                paramDict[key] = [vehicleModelChangeDict[key][i]]

            for n in range(numberEpisodes*numberRuns):
                paramDataframe = pd.concat([paramDataframe, pd.DataFrame(paramDict)])
            paramDataframe.to_csv(saveFilePathParams)
            
            self.executeRuns(numberEpisodes=numberEpisodes, numberRuns=numberRuns, render=render, training=False, saveDirectory=saveDirectory, saveFilename= saveFilenameData, vehicleModelParamDict=vehicleParamDict)



    def executeRuns(self, numberEpisodes, numberRuns, render, training, saveDirectory, saveFilename, vehicleModelParamDict):
        
        for run in range(numberRuns):
            print('Run: ', str(run))
            self.drivingAlgorithm.reset(run=run, training=training)
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

            self.resultsDataframe = pd.concat([self.resultsDataframe, pd.DataFrame([dataRow])])
            
            # # Update sliding averages over multiple episodes
            # averageReward = np.mean(episodesReward[-self.slidingWindow:])
            # averageProgress = np.mean(episodesProgress[-self.slidingWindow:])

            print(f"{'Run':4s} {run:2.0f} {'| Episode':8s} {episode:5.0f} {'| Reward: ':8s} {episodeReward:6.2f} {'| Progress: ':12s} {episodeProgress/self.trackLine.distance[-1]*100:7.2f} {'%':1s}  {'| Lap Time: ':12s} {lapTime:3.2f}  ")

            # print('Episode: ', str(episode), ' | ', dataRow)

            if ((episode%10==0) or (episode==numberEpisodes-1)) and (episode!=0):
                if training==True:
                    print("Agent was saved: ", self.conf.name)
                    self.drivingAlgorithm.agent.save_agent(name=self.conf.name, run=run)
                
                self.resultsDataframe.to_csv(filePath)
                print("Data was saved: ", self.conf.name)

    
    

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
            xStart, yStart, yawStart, velStart =  self.generateInitialPose()
            initialPose = np.array([[xStart, yStart, yawStart, velStart]])
        
        # Reset componenets of the environmnet (vehicle and trackline)
        obs, step_reward, done, info = self.env.reset(initialPose)
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
        while not done and (lapTime <= self.conf.trainEpisodeTimeLimit) and (episodeProgress <= self.trackLine.distance[-1]):
            

            if i % self.conf.planSteps == 0:
                plan = self.drivingAlgorithm.generatePlan(next_obs)

            if i % self.conf.controlSteps == 0:
                controlActions = np.array([self.drivingAlgorithm.generateControlAction(plan, next_obs)])

            next_obs, step_time, done, info = self.env.step(controlActions)
            
            if ((i+1) % self.learnSteps == 0 and i>1) or (done==True):
                
                # Get information for 1 [learning] time step 
                timeStepProgress = self.trackLine.findTimeStepProgress(next_obs)
                reward = self.rewardSignal.calculateReward(timeStepProgress, done)


                # Update information for entire episode
                episodeReward += reward
                episodeProgress += timeStepProgress
                
                if training == True:
                    self.drivingAlgorithm.storeTransition(obs, reward, next_obs, int(done))
                    self.drivingAlgorithm.learn()
                
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
        if done==False:
            episodeCrash=0
            lapTime=i/100

        if saveTrajectory==False:
            return episodeCrash, episodeReward, episodeProgress, lapTime
        elif saveTrajectory==True:
            return episodeCrash, episodeReward, episodeProgress, lapTime, episodeTrajectory, episodeProgresses
        


    def executeOneEpisodeLegacy(self, initialPose, saveTrajectory, training, render, vehicleModelParamDict):
        
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
            xStart, yStart, yawStart =  self.generateInitialPose()
            initialPose = np.array([[xStart, yStart, yawStart]])
        
        # Reset componenets of the environmnet (vehicle and trackline)
        obs, step_reward, done, info = self.env.reset(initialPose)
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
        while not done and (lapTime <= self.conf.trainEpisodeTimeLimit) and (episodeProgress <= self.trackLine.distance[-1]):
            

            if i % self.conf.planSteps == 0:
                plan = self.drivingAlgorithm.generatePlan(obs)

            if i % self.conf.controlSteps == 0:
                controlActions = np.array([self.drivingAlgorithm.generateControlAction(plan, obs)])

            next_obs, step_time, done, info = self.env.step(controlActions)
            
            if ((i+1) % self.learnSteps == 0 and i>1) or (done==True):
                
                # Get information for 1 [learning] time step 
                timeStepProgress = self.trackLine.findTimeStepProgress(next_obs)
                reward = self.rewardSignal.calculateReward(timeStepProgress, done)


                # Update information for entire episode
                episodeReward += reward
                episodeProgress += timeStepProgress
                
                if training == True:
                    self.drivingAlgorithm.storeTransition(obs, reward, next_obs, int(done))
                    self.drivingAlgorithm.learn()
                
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
        if done==False:
            episodeCrash=0
            lapTime=i/100

        if saveTrajectory==False:
            return episodeCrash, episodeReward, episodeProgress, lapTime
        elif saveTrajectory==True:
            return episodeCrash, episodeReward, episodeProgress, lapTime, episodeTrajectory, episodeProgresses

    





