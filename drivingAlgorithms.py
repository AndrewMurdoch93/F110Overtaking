import numpy as np


import agentTD3
import mapping
import pathTracker





class PartialEndtoEnd():
    
    """
    This class handles the interface between the agent and environment
    """
    
    def __init__(self, conf, trackLine):
        
        self.conf = conf
        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity
        self.LiDARRange = 30

        #still need to automate this
        self.m = mapping.map(conf.map_path)
        
        self.xOffset = self.m.origin[0]
        self.yOffset = self.m.origin[1]
        self.yMaximum = self.m.map_height
        self.xMaximum = self.m.map_width
        self.maxTheta = 2*np.pi 

        # Define components
        self.trackLine = trackLine
        self.steeringControl = pathTracker.purePursuit(conf)
       

    def reset(self, run, training):
        
        self.training=training
        self.agent = agentTD3.agent(self.conf)

        if not self.training:
            self.agent.load_weights(name = self.conf.name, run=run)


    def generateLocalPath(self, obs, nn_action):
        
        track_width = self.conf.trackWidth


        s_0_ind, s_0, n_0, theta = self.trackLine.convert_xy_obs_to_sn(obs)

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

        path = {'x': pathX,
                'y': pathY
        }

        return path
        

    
    def generatePlan(self, obs):
        """
        Partial end-to-end driving algorithm outputs a polynomial path in the Frenet frame.
        """

        nn_obs = self.transformObservation(obs)
        self.nn_action = self.agent.choose_action(nn_obs, self.training)
        path = self.generateLocalPath(obs, self.nn_action)
        velocity_ref = self.nn_action[1]*(self.maxVelocity-self.minVelocity)/2 + self.minVelocity + (self.maxVelocity-self.minVelocity)/2

        plan = {'path': path,
        'velocity': velocity_ref
        }

        return plan
    
    def generateControlAction(self, plan, obs):

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
        velocity_ref = plan['velocity']

        # Combine steering and velocity into an action
        controlAction = np.array([delta_ref, velocity_ref])

        return controlAction

    def transformObservation(self, obs):
        
        xObs = (obs['poses_x'][0]-self.xOffset)/(self.xMaximum)
        yObs = (obs['poses_y'][0]-self.yOffset)/(self.yMaximum)
        thetaObs = (obs['poses_theta'][0])/(self.maxTheta)
        vObs = (obs['linear_vels_x'][0])/(self.maxVelocity)

        pose = np.array([xObs, yObs, thetaObs, vObs])
        nn_observation = np.concatenate((pose, obs['scans'][0]/20.0))


        return nn_observation
    
    def storeTransition(self, obs, reward, next_obs, done):
        
        """
        Stores a transition tuple to agent replay buffer.
        Since main loop has action scaled to vehicle constraint values, nn_action is saved and accessed from drivingAlgorithm, 
        instead of passing it to function as parameter.
        """
        nn_obs = self.transformObservation(obs)
        nn_next_obs = self.transformObservation(next_obs)
        self.agent.store_transition(nn_obs, self.nn_action, reward, nn_next_obs, done)


    def learn(self):
        """
        Calls the learning function of the agent
        """
        self.agent.learn()

    
    




class EndtoEnd():
    
    """
    This class contains all methods pertaining to the end-to-end algorithm
    """
    
    def __init__(self, conf):
        
        self.conf = conf

        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity
        self.LiDARRange = 30
        
        self.m = mapping.map(conf.map_path)
        
        self.xOffset = self.m.origin[0]
        self.yOffset = self.m.origin[1]
        self.yMaximum = self.m.map_height
        self.xMaximum = self.m.map_width
        self.maxTheta = 2*np.pi 

        


    def reset(self, run, training):

        self.training=training
        self.agent = agentTD3.agent(self.conf)
        if not self.training:
            self.agent.load_weights(name = self.conf.name, run=run)


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
        """
        Takes observation form F1Tenth simulator (in the form of a dictionary), 
        and transforms the relevant information into a numpy array.
        """
        
        xObs = (obs['poses_x'][0]-self.xOffset)/(self.xMaximum)
        yObs = (obs['poses_y'][0]-self.yOffset)/(self.yMaximum)
        thetaObs = (obs['poses_theta'][0])/(self.maxTheta)
        vObs = (obs['linear_vels_x'][0])/(self.maxVelocity)

        pose = np.array([xObs, yObs, thetaObs, vObs])
        nn_observation = np.concatenate((pose, obs['scans'][0]/20.0))

        return nn_observation
    

    def generatePlan(self, obs):
        """
        End-to-end driving algorithm does not plan.
        """

        return None 
    

    def generateControlAction(self, plan, obs):
        """
        Plan is None, since End-to-end agent does not plan.
        Instead, the agent selects an action by forward passing obs through actor DNN.
        """
        nn_obs = self.transformObservation(obs)
        self.nn_action = self.agent.choose_action(nn_obs, self.training)
        controlAction = self.transformAction(self.nn_action)

        return controlAction
    

    def storeTransition(self, obs, reward, next_obs, done):
        
        """
        Stores a transition tuple to agent replay buffer.
        Since main loop has action scaled to vehicle constraint values, nn_action is saved and accessed from drivingAlgorithm, 
        instead of passing it to function as parameter.
        """
        nn_obs = self.transformObservation(obs)
        nn_next_obs = self.transformObservation(next_obs)
        self.agent.store_transition(nn_obs, self.nn_action, reward, nn_next_obs, done)


    def learn(self):
        """
        Calls the learning function of the agent
        """
        self.agent.learn()




class purePursuitCenterlineFollower():
    
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    """

    def __init__(self, conf, trackLine):
        
        self.conf = conf
        self.trackLine = trackLine
        self.steeringControl = pathTracker.purePursuit(conf)

        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity



    def reset(self, run, training=False):
        
        self.steeringControl.read_centerline_waypoints_csv()
        self.training=False


    def generatePlan(self, obs):
        """
        Pure pursuit centerline follower driving algorithm does not plan.
        """

        return None 
    

    def generateControlAction(self, plan, obs, idx):
        """
        Plan is None, since this algorithm only follows the predetermined centerline
        """
        target_index, _ = self.steeringControl.search_target_waypoint(obs['poses_x'][idx], obs['poses_y'][idx], obs['linear_vels_x'][idx])
        
        # Get desired steering angle based on target point
        delta_ref, target_index = self.steeringControl.pure_pursuit_steer_control(obs['poses_x'][idx], obs['poses_y'][idx], obs['poses_theta'][idx], obs['linear_vels_x'][idx], target_index)

        # Select velocity
        velocity_ref = 4

        # Combine steering and velocity into an action
        controlAction = [delta_ref, velocity_ref]
        
        return controlAction
    



class purePursuitCarFollower():
    
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    """

    def __init__(self, conf, trackLine):
        
        self.conf = conf
        self.trackLine = trackLine
        self.steeringControl = pathTracker.purePursuit(conf)

        self.maxVelocity = conf.maxVelocity
        self.minVelocity = conf.minVelocity



    def reset(self, run, training=False):
        
        self.steeringControl.read_centerline_waypoints_csv()
        self.training=False


    def generatePlan(self, obs):
        """
        Pure pursuit centerline follower driving algorithm does not plan.
        """

        return None 
    

    def generateControlAction(self, plan, obs, idx):
        """
        Plan is None, since this algorithm only follows the predetermined centerline
        """
        # target_index, _ = self.steeringControl.search_target_waypoint(obs['poses_x'][idx], obs['poses_y'][idx], obs['linear_vels_x'][idx])
        
        self.steeringControl.record_waypoints(cx = [obs['poses_x'][0]], cy=[obs['poses_y'][0]], cyaw=[obs['poses_theta'][0]])

        target_index = 0

        # Get desired steering angle based on target point
        delta_ref, target_index = self.steeringControl.pure_pursuit_steer_control(obs['poses_x'][idx], obs['poses_y'][idx], obs['poses_theta'][idx], obs['linear_vels_x'][idx], target_index)

        # Select velocity
        velocity_ref = 4

        # Combine steering and velocity into an action
        controlAction = [delta_ref, velocity_ref]
        
        return controlAction
    

