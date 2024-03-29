import numpy as np
import functions
from controllers import LQRController



class LQRLineFollower():
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    """

    def __init__(self, conf, line, vehicleNumber):
        
        self.conf = conf
        self.trackLine = line
        self.referenceVelocity = conf.referenceVelocity

        LQRConfig = functions.openConfigFile('drivingAlgorithms/controllers/'+conf.controllerConfig)

        self.steeringControl = LQRController.LQRSteer(LQRConfig, vehicleNumber)
        self.steeringControl.record_waypoints(cx=self.trackLine.cx, cy=self.trackLine.cy, cyaw=self.trackLine.cyaw, ck=self.trackLine.ccurve)
        self.vehicleNumber = vehicleNumber
        self.e = 0
        self.e_th = 0

    def reset(self, **kwargs):
        """
        reset method is called at the start of every episode
        """
        
        self.steeringControl.record_waypoints(cx=self.trackLine.cx, cy=self.trackLine.cy, cyaw=self.trackLine.cyaw, ck=self.trackLine.ccurve)
        self.timeStep = 0


    def stepDrivingAlgorithm(self, obs):
        """
        This method is called from the main simulation loop at every time step
        It calls the methods to generate a plan and control action at the correct interval
        Returns a control action to the main simulation loop
        """ 

        if self.timeStep % self.conf.controllerInterval == 0:
            self.controlAction = self.generateControlAction(obs)
        
        self.timeStep += 1

        return self.controlAction
    

    def generateControlAction(self, obs):
        """
        Plan is None, since this algorithm only follows the predetermined line.
        Control action steers the car towards a target point ahead of the car on the line.
        """

        delta_ref, target_ind, self.e, self.e_th = self.steeringControl.lqr_steering_control(obs=obs, pe=self.e, pth_e=self.e_th)

        # Select velocity - for now, it's just constant
        velocity_ref = self.conf.referenceVelocity

        # Combine steering and velocity into an action
        controlAction = [delta_ref, velocity_ref]
        
        return controlAction
    

