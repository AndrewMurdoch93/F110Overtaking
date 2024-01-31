import numpy as np

import agentTrainer
import display


vehicleModelParamDict = {'mu': 1.04,
               'C_Sf': 4.718,
               'C_Sr': 5.4562,
               'lf': 0.15875,
               'lr': 0.17145,
               'h': 0.074,
               'm': 3.74,
               'I': 0.04712,
               's_min': -0.4189,
               's_max': 0.4189,
               'sv_min': -3.2,
               'sv_max': 3.2,
               'v_switch':7.319,
               'a_max': 9.51,
               'v_min':-5.0,
               'v_max': 20.0,
               'width': 0.31,
               'length': 0.58
               }


vehicleModelChangeDict = {
    'mu': [0.6, 0.7, 0.8, 0.9, 1]
}


if __name__=='__main__':


    # agentTrainer.trainAgent('configPartialEndtoEndAgent', render=False)
    # agentTrainer.trainAgent('configTD3Agent', render=False)
    # agentTrainer.trainAgent('partialEndtoEndAgentPorto', render=False)
    # agentTrainer.trainAgent('endtoEndAgentPorto', render=False)
    # agentTrainer.trainAgent('partialEndtoEndAgentBarcelona1', render=False)
    # agentTrainer.trainAgent('endtoEndAgentBarcelona1', render=False)
    # agentTrainer.trainAgent('partialEndtoEndAgentBarcelona2', render=False)
    # agentTrainer.trainAgent('endtoEndAgentBarcelona2', render=False)
    # agentTrainer.trainAgent('testAgentBarcelona', render=True)


    # agentTrainer.trainEvaluate(configFileName='endtoEndAgentBarcelona1', numberTrainEpisodes=3000, numberEvalEpisodes=10, evalInterval=50, render=False)
    # agentTrainer.trainEvaluate(configFileName='partialEndtoEndAgentBarcelona1', numberTrainEpisodes=1000, numberEvalEpisodes=10, evalInterval=50, render=False)
    # agentTrainer.trainEvaluate(configFileName='endtoEndAgentBarcelona2', numberTrainEpisodes=3000, numberEvalEpisodes=10, evalInterval=50, render=False)
    # agentTrainer.trainEvaluate(configFileName='partialEndtoEndAgentBarcelona2', numberTrainEpisodes=1000, numberEvalEpisodes=10, evalInterval=50, render=False)

    # agentTrainer.testAgent('configTD3Agent', render=False, episodes=100)
    # agentTrainer.testAgent('configPartialEndtoEndAgent', render=False, episodes=100)
    # agentTrainer.testAgent('partialEndtoEndAgentPorto', render=True, episodes=100)
    agentTrainer.testAgent('endtoEndAgentPorto', render=True, episodes=100)
    # agentTrainer.testAgent('partialEndtoEndAgentBarcelona', render=False, episodes=10)
    # agentTrainer.testAgent('endtoEndAgentBarcelona', render=False, episodes=10)
    # agentTrainer.testAgent('testAgentBarcelona', render=True)

    # agentTrainer.testAgent('partialEndtoEndAgentBarcelona2', render=False, episodes=50)
    # agentTrainer.testAgent('endtoEndAgentBarcelona2', render=False, episodes=50)

    # agentTrainer.modelMismatchExperiment(configFileName='configTD3Agent', saveFilename='configTD3Agentmu', vehicleModelParamDict=vehicleModelParamDict,  vehicleModelChangeDict=vehicleModelChangeDict, render=False, episodes=50)
    # agentTrainer.modelMismatchExperiment(configFileName='configPartialEndtoEndAgent', saveFilename='configPartialEndtoEndAgentmu', vehicleModelParamDict=vehicleModelParamDict,  vehicleModelChangeDict=vehicleModelChangeDict, render=False, episodes=50)
    # agentTrainer.modelMismatchExperiment(configFileName='partialEndtoEndAgentPorto', saveFilename='partialEndtoEndAgentPorto_mu', vehicleModelParamDict=vehicleModelParamDict,  vehicleModelChangeDict=vehicleModelChangeDict, render=False, episodes=100)
    # agentTrainer.modelMismatchExperiment(configFileName='endtoEndAgentPorto', saveFilename='endtoEndAgentPorto_mu', vehicleModelParamDict=vehicleModelParamDict,  vehicleModelChangeDict=vehicleModelChangeDict, render=False, episodes=100)


    # display.plotTrainingCurves(['configTD3Agent', 'configPartialEndtoEndAgent'], labels = ['End-to-end', 'Partial end-to-end'])
    # display.plotTrainingCurves(['endtoEndAgentPorto', 'partialEndtoEndAgentPorto'], labels = ['End-to-end', 'Partial end-to-end'])
    # display.plotTrainingCurves(['endtoEndAgentBarcelona', 'partialEndtoEndAgentBarcelona'], labels = ['End-to-end', 'Partial end-to-end'])
    # display.plotTrainingCurves(['endtoEndAgentBarcelona2', 'partialEndtoEndAgentBarcelona2'], labels = ['End-to-end', 'Partial end-to-end'])


    # display.plotEvaluationRun(configFileNames=['partialEndtoEndAgentPorto'], labels=['partial end-to-end'])
    # display.plotEvaluationRun(configFileNames=['endtoEndAgentBarcelona1'], labels=['end-to-end'])
    # display.plotEvaluationRun(configFileNames=['endtoEndAgentBarcelona2', 'partialEndtoEndAgentBarcelona2'], labels=['end-to-end', 'partial end-to-end'])
    
    # display.plotEvaluationRun(configFileNames=['endtoEndAgentBarcelona1', 'partialEndtoEndAgentBarcelona1'], labels=['end-to-end', 'partial end-to-end'])
    # display.plotEvaluationRun(configFileNames=['endtoEndAgentBarcelona2', 'partialEndtoEndAgentBarcelona2'], labels=['end-to-end', 'partial end-to-end'])

    # display.testResults('configTD3Agent')
    # display.testResults('configPartialEndtoEndAgent')

    # display.testResults('endtoEndAgentPorto')
    # display.testResults('partialEndtoEndAgentPorto')
    # display.testResults('endtoEndAgentBarcelona2')
    # display.testResults('partialEndtoEndAgentBarcelona')


    # display.saveOneLap(configFileName='configTD3Agent', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end')
    # display.saveOneLap(configFileName='configPartialEndtoEndAgent', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end')
    # display.saveOneLap(configFileName='endtoEndAgentPorto', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end_mu')
    # display.saveOneLap(configFileName='partialEndtoEndAgentPorto', run=1, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end_mu')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 0')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona', run=1, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 1')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona', run=2, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 2')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona', run=3, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 3')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona', run=4, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 4')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona 0')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=1, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona 1')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=2, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona 2')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=3, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona 3')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=4, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona 4')
    # display.saveOneLap(configFileName='testAgentBarcelona', run=0, initialPose=np.array([[0., 0., 0., 3.0]]), vehicleModelParamDict=vehicleModelParamDict, fileName='test')
    


    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona1', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona 5')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona2', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='partial end-to-end barcelona eval')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona2', run=0, initialPose=np.array([[0., 0., 0.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='end-to-end barcelona eval')
    # display.saveOneLap(configFileName='endtoEndAgentBarcelona1', run=0, initialPose=np.array([[0., 0., 0., 3.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='ete eval')
    # display.saveOneLap(configFileName='partialEndtoEndAgentBarcelona1', run=0, initialPose=np.array([[0., 0., 0., 3.]]), vehicleModelParamDict=vehicleModelParamDict, fileName='pete eval')
    


    # display.displayLap(fileNames=['end-to-end', 'partial end-to-end'], labels=['End-to-end', 'Partial end-to-end'])
    # display.displayLap(fileNames=['end-to-end_mu', 'partial end-to-end_mu'], labels=['End-to-end', 'Partial end-to-end'])
    # display.displayLap(fileNames=['end-to-end barcelona', 'partial end-to-end barcelona'], labels=['end-to-end', 'Partial end-to-end'])


    # display.displayLap(fileNames=['partial end-to-end barcelona 0', 'partial end-to-end barcelona 1', 
    #                       'partial end-to-end barcelona 2', 'partial end-to-end barcelona 3', 
    #                       'partial end-to-end barcelona 4'], labels=['run 0', 'run 1', 'run 2', 'run 3', 'run 4'])

    # display.displayLap(fileNames=['end-to-end barcelona 0', 'end-to-end barcelona 1', 
    #                       'end-to-end barcelona 2', 'end-to-end barcelona 3', 
    #                       'end-to-end barcelona 4'], labels=['run 0', 'run 1', 'run 2', 'run 3', 'run 4'])


    # display.displayLap(fileNames=['ete eval', 'pete eval'], labels=['ETE', 'PETE'])
    # display.displayLap(fileNames=['test'], labels=['PETE'])


    # display.plotModelMismatch(filenames=['configTD3Agentmu', 'configPartialEndtoEndAgentmu'], runs=[2,0], labels=['End-to-end', 'Partial end-to-end'])
    # display.plotModelMismatch(filenames=['endtoEndAgentPorto_mu', 'partialEndtoEndAgentPorto_mu'], runs=[2,1], labels=['End-to-end', 'Partial end-to-end'])