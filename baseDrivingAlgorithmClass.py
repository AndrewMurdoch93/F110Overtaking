class DrivingAlgorithm():
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    Specific driving algorithms are inherit form this class
    """
    
    def __init__(self, vehicleNumber=0, perceptionDT=0.01, plannerDT=0.01, controllerDT=0.01):
        """
        Initialises arguments for driving algorithm
        """
        
        # Initialise values for arguments
        self.vehicleNumber = vehicleNumber
        self.perceptionDT = perceptionDT
        self.plannerDT = plannerDT
        self.controllerDT = controllerDT

        # Initialise outputs of perception, planner and controller
        self.observation = None
        self.plan = None
        self.controlAction = [0, 0]

    def reset(self, **kwargs):
        """
        Reset method is called at the start of every episode
        Implementation can be overrided by child class
        """
        self.timeStep = 0


    def stepDrivingAlgorithm(self, obs):
        """
        This method is called from the main simulation loop at every time step
        Receives an observation from F1tenth simulator
        Returns a control action (list of length 2) to the main simulation loop
        """ 
        
        if (self.timeStep % int(1/self.perception.DT) == 0):
            self.observation = self.perception(obs)

        if (self.timeStep % int(1/self.planner.DT) == 0):
            self.plan = self.planner()

        if (self.timeStep % int(1/self.controller.DT) == 0):
            self.controlAction = self.controller()     

        self.timeStep += 1

        return self.controlAction

    
    def perception(self, obs):
        """
        Default observation method returns observation from simulator
        """
        return obs
    
    
    def planner(self):
        """
        Returns a plan, typically a list of tuples [(x1,y1,v1), (x2,y2,v2)]...
        But details are specific to each algorithm

        Default planner returns no plan (None)
        Implementation should be overrided in child class
        """
        return None
    

    def controller(self):
        """
        Returns a list [steering angle, velocity]
        
        Default controller returns control corresponding to
        steering angle = 0 deg, velocity = 0 m/s
        """
        return [0,0]


    def recordAlgorithmInformation(self):
        """
        Method is called to generate a dictionary containing information about the driving algorithm
        Implementation should be overridden by child class
        """
        dictionary = {}

        return dictionary 


if __name__ == '__main__':
    drivingAlgorithm = DrivingAlgorithm()