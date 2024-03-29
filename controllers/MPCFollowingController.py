from re import X
import cvxpy
import math
import numpy as np
import sys
import os
import functions
import matplotlib.pyplot as plt
from scipy.spatial import distance


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None



class MPC():

    def __init__(self, controllerConf, vehicleConf, vehicleNumber, referenceVelocity):

        self.controllerConf = controllerConf
        self.vehicleConf = vehicleConf
        self.vehicleNumber = vehicleNumber
        self.state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
        self.targetState = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
        self.dl = 0.1   # Default parameter for cubic spline course

        # MPC parameters
        self.NX = self.controllerConf.NX   # X (states) = x, y, v, yaw
        self.NU = self.controllerConf.NU   # U (controls) = [accel, steer]
        self.T = self.controllerConf.T     # horizon length
        self.DU_TH = self.controllerConf.DU_TH          # iteration finish param
        self.MAX_ITER = self.controllerConf.MAX_ITER    # Max iteration
        self.DT = self.controllerConf.DT                # [s] time tick
        self.N_IND_SEARCH = self.controllerConf.N_IND_SEARCH
        self.R = np.diag([self.controllerConf.R_accel, self.controllerConf.R_steer])   # input cost matrix
        self.Rd = np.diag([self.controllerConf.Rd_accel, self.controllerConf.Rd_steer])   # input difference cost matrix
        self.Q = np.diag([self.controllerConf.Q_x, self.controllerConf.Q_y, self.controllerConf.Q_v, self.controllerConf.Q_yaw]) # state cost matrix
        self.Qf = self.Q
        self.TARGET_SPEED = referenceVelocity

        # Vehicle model parameters
        self.MAX_STEER = self.vehicleConf.s_max
        self.WB = self.vehicleConf.lf + self.vehicleConf.lr
        self.MAX_SPEED = 4
        self.MIN_SPEED = self.vehicleConf.v_min
        self.MAX_DSTEER = self.vehicleConf.sv_max
        self.MAX_ACCEL = self.vehicleConf.a_max
        # self.MAX_ACCEL = 10

        self.obstacle = np.zeros((2,6))
        self.obstacle[0,:] = 2

    def reset(self, trackLine):

        self.record_waypoints(cx=trackLine.cx, cy=trackLine.cy, cyaw=trackLine.cyaw, ck=trackLine.ccurve)
        self.calc_speed_profile(target_speed=self.TARGET_SPEED)
        
        self.target_ind = 0
        self.odelta, self.oa = None, None


    def record_waypoints(self, cx, cy, cyaw, ck):
    
        self.cx=cx
        self.cy=cy
        self.cyaw_smooth = self.smooth_yaw(cyaw)
        self.cyaw = cyaw

        # cyaw_new = cyaw
        # for idx, yaw in enumerate(cyaw):
        #     cyaw_new[idx] = functions.pi_2_pi(yaw)
        # self.cyaw = cyaw_new

        # plt.plot(self.cyaw)
        # plt.show()

        self.ck = ck


    def smooth_yaw(self, yaw):

        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.MAX_STEER:
            delta = self.MAX_STEER
        elif delta <= -self.MAX_STEER:
            delta = -self.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.WB * math.tan(delta) * self.DT
        state.v = state.v + a * self.DT

        if state.v > self.MAX_SPEED:
            state.v = self.MAX_SPEED
        elif state.v < self.MIN_SPEED:
            state.v = self.MIN_SPEED

        return state


    def get_linear_model_matrix(self, v, phi, delta):

        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = -self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta)/self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = - self.DT * v * math.cos(phi) * phi
        C[3] = - self.DT * v * delta / ( self.WB * math.cos(delta) ** 2)

        return A, B, C


    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()


    def predict_motion(self, x0, oa, od, xref):
        
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar

    
    
    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC control with updating operational point iteratively
        """

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov

    
    def linear_mpc_control(self, xref, xbar, x0, dref):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            # if t != 0:
            #     cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]


            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                self.MAX_DSTEER * self.DT]

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)



        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        


        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = self.get_nparray_from_matrix(x.value[0, :])
            oy = self.get_nparray_from_matrix(x.value[1, :])
            ov = self.get_nparray_from_matrix(x.value[2, :])
            oyaw = self.get_nparray_from_matrix(x.value[3, :])
            oa = self.get_nparray_from_matrix(u.value[0, :])
            odelta = self.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov



    def calc_speed_profile(self, target_speed):

        speed_profile = [target_speed] * len(self.cx)
        direction = 1.0  # forward

        # Set stop point
        for i in range(len(self.cx) - 1):
            dx = self.cx[i + 1] - self.cx[i]
            dy = self.cy[i + 1] - self.cy[i]

            move_direction = math.atan2(dy, dx)

            if dx != 0.0 and dy != 0.0:
                dangle = abs(functions.pi_2_pi(move_direction - self.cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

        speed_profile[-1] = 0.0

        self.sp = speed_profile


    def calc_nearest_index(self, state, pind):

        if pind==0:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
        else:
            dx = [state.x - icx for icx in self.cx[pind:(pind + self.N_IND_SEARCH)]]
            dy = [state.y - icy for icy in self.cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind)

        dxl = self.cx[ind] - state.x
        dyl = self.cy[ind] - state.y

        angle = functions.pi_2_pi(self.cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def transformObsToState(self, obs, vehicleNumber):
        
        state = State(
                    x = obs['poses_x'][vehicleNumber],
                    y = obs['poses_y'][vehicleNumber],
                    # yaw = functions.pi_2_pi(obs['poses_theta'][self.vehicleNumber]),
                    yaw = obs['poses_theta'][vehicleNumber],
                    # yaw = self.correctYaw(obs['poses_theta'][self.vehicleNumber]),
                    v = obs['linear_vels_x'][vehicleNumber]
        )

        return state

    def correctYaw(self, yaw):
        
        ns = np.arange(-10,10,1)
        yaws = yaw + ns*2.0*np.pi
        error = self.cyaw[self.target_ind]-yaws
        squaredError = np.square(error)
        ind = np.argmin(squaredError)
        n = ns[ind]
        newYaw = yaw + n*2.0*np.pi
        return newYaw


    def getDistanceVector(self, x, y):

        distances = np.zeros(len(x))
        
        for i in range(len(distances)):
            distances[i] = np.sqrt((x[i]**2 + y[i]**2 ))

        return distances
    
    def getreferenceState(self, state, targetState):
        """
        Calculate the state that the ego vehicle is aiming for
        also get dref, operational steer point
        """
        
        # Create variables
        lineBetweenVehicles = np.zeros((2,10))

        lineBetweenVehicles[0,:] = np.linspace(state.x, targetState.x, 10)
        lineBetweenVehicles[1,:] = np.linspace(state.y, targetState.y, 10)
        
        # distances = self.getDistanceVector(lineBetweenVehicles[0,:], lineBetweenVehicles[1,:])
        # targetDistance = distances[-1]-0.2
        # targetIndex = (np.abs(distances - targetDistance)).argmin()
        # x = lineBetweenVehicles[0,targetIndex]
        # y = lineBetweenVehicles[1,targetIndex]

        x = targetState.x
        y = targetState.y
        v = targetState.v
        yaw = targetState.yaw


        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))


        for i in range(self.T + 1):
            xref[0, i] = x # x
            xref[1, i] = y # y
            xref[2, i] = v # v
            xref[3, i] = yaw # yaw
            dref[0, i] = 0.0
        

        return xref, dref




    def getAction(self, obs):
        
        self.state = self.transformObsToState(obs, self.vehicleNumber)
        self.targetState = self.transformObsToState(obs=obs, vehicleNumber=1)

        self.target_ind, _ = self.calc_nearest_index(self.state, self.target_ind)

        self.state.yaw = self.correctYaw(self.state.yaw)
        self.targetState.yaw = self.correctYaw(self.targetState.yaw)

        self.xref, self.dref = self.getreferenceState(self.state, self.targetState)

        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state 
        self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov = self.iterative_linear_mpc_control(self.xref, x0, self.dref, self.oa, self.odelta)
        
        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]
            vi = self.state.v+ai*self.DT


        plt.figure(1, figsize=(5,4))
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(self.cx, self.cy, '-')
        plt.plot(self.xref[0], self.xref[1], 'o')
        plt.plot(self.ox, self.oy, 's')
        plt.plot(self.state.x, self.state.y, 'x')
        plt.plot(self.targetState.x, self.targetState.y, '+')
        plt.legend(['Trackline', 'Reference trajectory', 'Planned trajectory', 'Ego vehicle position', 'Target vehicle position'])
        plt.axis('scaled')
        # plt.ylim(-1,1)
        plt.pause(0.0001)

        return di, vi