import numpy as np
import functions
import math
from controllers import MPCController




class MPCLineFollower():
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    """

    def __init__(self, algorithmConf, vehicleConf, line, vehicleNumber):
        
        self.algorithmConf = algorithmConf
        self.vehicleConf = vehicleConf
        self.trackLine = line
 

        self.controllerConf = functions.openConfigFile('drivingAlgorithms/controllers/'+algorithmConf.controllerConfig)
        
        self.controller = MPCController.MPC(self.controllerConf, vehicleConf, vehicleNumber)
        
        # self.controller.record_waypoints(cx=self.trackLine.cx, cy=self.trackLine.cy, cyaw=self.trackLine.cyaw, ck=self.trackLine.ccurve)
        # self.controller.calc_speed_profile(target_speed=self.controllerConf.TARGET_SPEED)



        self.vehicleNumber = vehicleNumber
        self.e = 0
        self.e_th = 0


    def reset(self, **kwargs):
        """
        reset method is called at the start of every episode
        """

        self.controller.reset(self.trackLine)
        self.timeStep = 0


    def stepDrivingAlgorithm(self, obs):
        """
        This method is called from the main simulation loop at every time step
        It calls the methods to generate a plan and control action at the correct interval
        Returns a control action to the main simulation loop
        """ 

        if self.timeStep % self.controllerConf.DT*10 == 0:
            self.controlAction = self.generateControlAction(obs)
        
        self.timeStep += 1

        return self.controlAction
    

    def generateControlAction(self, obs):
        """
        Plan is None, since this algorithm only follows the predetermined line.
        Control action steers the car towards a target point ahead of the car on the line.
        """

        delta_ref, velocity_ref = self.controller.getAction(obs)

        # Combine steering and velocity into an action
        controlAction = [delta_ref, velocity_ref]
        
        return controlAction



