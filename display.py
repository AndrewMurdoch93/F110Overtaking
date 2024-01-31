


import agentTrainer
import functions
import mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import sys



def plotTrainingCurves(configFileNames, labels):

    dataframes = []
    legendTitle = ''


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(5.5,2.3))
    
    for configFileName in configFileNames:
        conf = functions.openConfigFile(configFileName)
    
        resultsDataFrame = openResultsFile(name = conf.name, training=True)
        averagedResultsDataFrame = averageTrainingResultsOverRuns(resultsDataFrame)
        slidingWindowAverageDataFrame = slidingWindowAverage(averagedResultsDataFrame)

        x = np.arange(len(slidingWindowAverageDataFrame['episodeReward']))

        plot(ax1, x, slidingWindowAverageDataFrame['episodeReward'], title='', xlabel='Episode', ylabel='Reward')
        plot(ax2, x, slidingWindowAverageDataFrame['episodeLapTime'], title='', xlabel='Episode', ylabel='Lap time [s]')
        plot(ax3, x, slidingWindowAverageDataFrame['episodeCrash']*100, title='', xlabel='Episode', ylabel='Failed laps [%]')

    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    plt.figlegend(labels=labels, title=legendTitle, loc='lower center', ncol=4, borderpad=1)
    plt.show()


def plotEvaluationRun(configFileNames, labels):
    dataframes = []
    legendTitle = ''


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(5.5,2.3))

    for configFileName in configFileNames:
        conf = functions.openConfigFile(configFileName)
        dataframe = pd.read_csv('experimentsData/evaluationData/' + conf.name + '.csv')

        averagedResultsDataFrame = dataframe.groupby('trainEpisode').mean()
        averagedResultsDataFrame = averagedResultsDataFrame.drop(['Unnamed: 0', 'evalEpisode'], axis=1)
    

        x = list(averagedResultsDataFrame.index)

        plot(ax1, x, averagedResultsDataFrame['episodeReward'], title='', xlabel='Episode', ylabel='Reward')
        plot(ax2, x, averagedResultsDataFrame['episodeLapTime'], title='', xlabel='Episode', ylabel='Lap time [s]')
        plot(ax3, x, averagedResultsDataFrame['episodeCrash']*100, title='', xlabel='Episode', ylabel='Failed laps [%]')


    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    plt.figlegend(labels=labels, title=legendTitle, loc='lower center', ncol=4, borderpad=1)
    plt.show()






def plotModelMismatch(filenames, runs, labels):

    legendTitle = ''


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



    fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(5.5,2.5))
    
    for run, filename in zip(runs, filenames):
        data = pd.read_csv('experimentsData/modelMismatchData/' + filename + 'Data.csv')
        params = pd.read_csv('experimentsData/modelMismatchData/' + filename + 'Params.csv')
        results = pd.concat([params, data], axis=1).drop('Unnamed: 0', axis=1)
        results['run'] =  results['run'].astype(int)
        crashes = pd.DataFrame(results.groupby(['mu','run'])['episodeCrash'].mean())
        
        x = list(crashes.xs(run,level='run').index)
        y = np.array(crashes.xs(run,level='run')['episodeCrash'])*100

        plot(ax1, x, y, title='', xlabel='True friction coefficient', ylabel='Failed laps [%]')
      

    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    plt.figlegend(labels=labels, title=legendTitle, loc='lower center', ncol=4, borderpad=1)
    plt.show()






def openResultsFile(name, training):
    
    if training==True:
        directory = 'experimentsData/trainingData'
    if training==False:
        directory = 'experimentsData/testingData'

    filePath = sys.path[0] + '/' + directory + '/' + name + '.csv'

    resultsDataframe = pd.read_csv(filePath)

    return resultsDataframe


def averageTrainingResultsOverRuns(resultsDataframe):

    averagedResultsDataFrame = resultsDataframe.groupby('episode').mean()

    return averagedResultsDataFrame



def slidingWindowAverage(resultsDataframe):

    
    # slidingWindowAverageDataFrame = slidingWindowAverageColumn(resultsDataframe['Lap time'])
    
    slidingWindowAverageDataframe = resultsDataframe
    for col in slidingWindowAverageDataframe.columns:
        slidingWindowAverageDataframe[col] = slidingWindowAverageColumn(slidingWindowAverageDataframe[col])

    return slidingWindowAverageDataframe



def slidingWindowAverageColumn(column):
    
    window = 50

    column = np.array(column)
    averagedColumn = np.zeros(len(column))

    for i in range(len(column)):
        if i<window:
            x=0
        else:
            x=i-window
        averagedColumn[i] = np.nanmean(column[x:i+1])

    return averagedColumn
        




def plot(axis, x, y, title, xlabel, ylabel):

    axis = axisStyle(axis, title=title, xlabel=xlabel, ylabel=ylabel)
    axis.plot(x, y)




def axisStyle(axis, title, xlabel, ylabel):
    
    frameColor = 'black'
    gridColor = 'grey'

    axis.spines['bottom'].set_color(frameColor)
    axis.spines['top'].set_color(frameColor)
    axis.spines['left'].set_color(frameColor)
    axis.spines['right'].set_color(frameColor)

    axis.grid(gridColor)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    axis.tick_params(axis=u'both', which=u'both',length=0)

    return axis


def testResults(configFileName):
    
    conf = functions.openConfigFile(configFileName)
    resultsDataFrame = pd.read_csv('experimentsData/testingData/'+conf.name+'.csv')
    
    groupedResultsDataFrame = resultsDataFrame.groupby('run').mean()
    groupedResultsDataFrame = groupedResultsDataFrame.drop(['Unnamed: 0', 'episode'], axis=1) 
    
    averagedResultsDataFrame = resultsDataFrame.mean()
    averagedResultsDataFrame = averagedResultsDataFrame.drop(['Unnamed: 0', 'episode', 'run']) 


    print("\nAgent: " + conf.name)
    print("\nResults per run: ")
    print(groupedResultsDataFrame)
    print("\nAverage results over all runs: ")
    print(averagedResultsDataFrame)



def displayLap(fileNames, labels):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(3,4))


    for fileName in fileNames:
        
        filePath = sys.path[0] + '/experimentsData/lapData/' + fileName + '.csv'
        lapDataframe = pd.read_csv(filePath)
        
        ax1.plot(lapDataframe['x'],  lapDataframe['y'])
        
        x = lapDataframe['progress']
        plot(axis=ax2, x=x, y=lapDataframe['velocity'], title='', xlabel='Progress along centerline [%]', ylabel='Velocity')

    plotTrack(ax1, lapDataframe['map'][0])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    plt.figlegend(labels=labels, title='', loc='lower center', ncol=4, borderpad=1)
    plt.show()




def plotTrack(axis, map):
    
    axis.axis('off')
    track = mapping.map(map)
    axis.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(track.origin[0], track.origin[0] + track.map_width, track.origin[1], track.origin[1]+track.map_height), cmap="gray")




def saveOneLap(configFileName, run, initialPose, vehicleModelParamDict, fileName):
    """
    Executes one lap.
    Saves the trajectory taken by the agent to a csv file.
    """
    conf = functions.openConfigFile(configFileName)
    agent_trainer = agentTrainer.AgentTrainer(conf=conf)
    agent_trainer.drivingAlgorithm.reset(run, training=False)
    episodeCrash, episodeReward, episodeProgress, lapTime, episodeTrajectory, episodeProgresses = agent_trainer.executeOneEpisode(initialPose=initialPose, saveTrajectory=True, training=False, render=False, vehicleModelParamDict=vehicleModelParamDict)

    trajectoryLen = len(episodeTrajectory)
    x = np.zeros(trajectoryLen)
    y = np.zeros(trajectoryLen)
    theta = np.zeros(trajectoryLen)
    velocity = np.zeros(trajectoryLen)

    for i in range(trajectoryLen):
        x[i] = episodeTrajectory[i][0]['poses_x'][0]
        y[i] = episodeTrajectory[i][0]['poses_y'][0] 
        theta[i] = episodeTrajectory[i][0]['poses_theta'][0] 
        velocity[i] = episodeTrajectory[i][0]['linear_vels_x'][0]
    
    trajectoryDict = {
        'map': conf.map_path,
        'x': x,
        'y': y,
        'theta': theta,
        'velocity': velocity,
        'progress': episodeProgresses/agent_trainer.trackLine.distance[-1]*100
    }
    df = pd.DataFrame(trajectoryDict)

    directory = 'experimentsData/lapData'
    filePath = sys.path[0] + '/' + directory + '/' + fileName + '.csv'
    
    df.to_csv(filePath)