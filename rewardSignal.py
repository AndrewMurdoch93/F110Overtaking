
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


