import cvxpy
import math
import numpy as np
import sys
import os
import functions


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

    def update_state(state, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * DT
        state.y = state.y + state.v * math.sin(state.yaw) * DT
        state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
        state.v = state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED
        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state


    def get_linear_model_matrix(self, v, phi, delta):

        A = np.zeros((self.controllerConf.NX, self.controllerConf.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.controllerConf.DT * math.cos(phi)
        A[0, 3] = -self.controllerConf.DT * v * math.sin(phi)
        A[1, 2] = self.controllerConf.DT * math.sin(phi)
        A[1, 3] = self.controllerConf.DT * v * math.cos(phi)
        A[3, 2] = self.controllerConf.DT * math.tan(delta) /   self.vehicleNumber.WB

        B = np.zeros((self.controllerConf.NX, self.controllerConf.NU))
        B[2, 0] = self.conf.DT
        B[3, 1] = self.conf.DT * v / (self.conf.WB * math.cos(delta) ** 2)

        C = np.zeros(self.controllerConf.NX)
        C[0] = self.controllerConf.DT * v * math.sin(phi) * phi
        C[1] = - self.controllerConf.DT * v * math.cos(phi) * phi
        C[3] = - self.controllerConf.DT * v * delta / (  self.vehicleNumber.WB * math.cos(delta) ** 2)

        return A, B, C


    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def predict_motion(self, x0, oa, od, xref):
        
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.controllerConf.T + 1)):
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
            oa = [0.0] * self.controllerConf.T
            od = [0.0] * self.controllerConf.T

        for i in range(self.controllerConf.MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.controllerConf.DU_TH:
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

        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))

        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, T] - x[:, t], Q)

            A, B, C = get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            # constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                MAX_DSTEER * DT]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov