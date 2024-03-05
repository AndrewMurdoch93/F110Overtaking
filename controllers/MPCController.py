import cvxpy
import math
import numpy as np
import sys
import os
import functions
import matplotlib.pyplot as plt


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

    def __init__(self, controllerConf, vehicleConf, vehicleNumber):

        self.controllerConf = controllerConf
        self.vehicleConf = vehicleConf
        self.vehicleNumber = vehicleNumber
        self.state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
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


        # Vehicle model parameters
        self.MAX_STEER = self.vehicleConf.s_max
        self.WB = self.vehicleConf.lf + self.vehicleConf.lr
        self.MAX_SPEED = self.vehicleConf.v_max
        self.MIN_SPEED = self.vehicleConf.v_min
        self.MAX_DSTEER = self.vehicleConf.sv_max
        self.MAX_ACCEL = self.vehicleConf.a_max

    def reset(self, trackLine):

        self.record_waypoints(cx=trackLine.cx, cy=trackLine.cy, cyaw=trackLine.cyaw, ck=trackLine.ccurve)
        self.calc_speed_profile(target_speed=self.controllerConf.TARGET_SPEED)
        
        self.target_ind = None
        self.odelta, self.oa = None, None



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
        A[3, 2] = self.DT * math.tan(delta) /   self.WB

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
        MPC contorl with updating operational point iteraitvely
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

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            # constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

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


    def record_waypoints(self, cx, cy, cyaw, ck):
    
        self.cx=cx
        self.cy=cy
        self.cyaw = self.smooth_yaw(cyaw)
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


    def calc_ref_trajectory(self, state, pind):

        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        ncourse = len(self.cx)

        ind, _ = self.calc_nearest_index(state, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(state.v) * self.DT
            dind = int(round(travel / self.dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self.cx[ind + dind] # x
                xref[1, i] = self.cy[ind + dind] # y
                xref[2, i] = self.sp[ind + dind] # v
                xref[3, i] = self.cyaw[ind + dind] # yaw
                dref[0, i] = 0.0
            else:
                xref[0, i] = self.cx[ncourse - 1] # x
                xref[1, i] = self.cy[ncourse - 1] # y
                xref[2, i] = self.sp[ncourse - 1] # v
                xref[3, i] = self.cyaw[ncourse - 1] # yaw
                dref[0, i] = 0.0

        return xref, ind, dref


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


    def transformObsToState(self, obs):
        
        state = State(
                    x = obs['poses_x'][self.vehicleNumber],
                    y = obs['poses_y'][self.vehicleNumber],
                    yaw = obs['poses_theta'][self.vehicleNumber],
                    v = obs['linear_vels_x'][self.vehicleNumber]
        )

        return state


    def getAction(self, obs):
        
        self.state = self.transformObsToState(obs)

        if self.target_ind==None:
             self.target_ind, _ = self.calc_nearest_index(self.state, 0)

        self.xref, self.target_ind, self.dref = self.calc_ref_trajectory(self.state, self.target_ind)

        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state 
        self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov = self.iterative_linear_mpc_control(self.xref, x0, self.dref, self.oa, self.odelta)
        
        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]
            vi = self.state.v+ai*self.DT


        plt.plot(self.cx, self.cy, '-')
        plt.plot(self.xref[0], self.xref[1], 'o')
        plt.plot(self.ox, self.oy, 's')
        plt.plot(self.state.x, self.state.y, 'x')
        plt.legend(['Trackline', 'Reference trajectory', 'Planned trajectory', 'Ego vehicle position'])
        plt.show()

    

        return di, vi