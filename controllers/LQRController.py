import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import pyglet

import functions



# LQR parameter
lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]

class State():
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


class LQRSteer():
    """
    This class contains all methods pertaining to a simple pure pusuit centerline following algorithm
    """

    def __init__(self, conf, vehicleNumber):
        
        # Parameters for initialisation
        self.conf = conf
        self.vehicleNumber = vehicleNumber
        self.old_nearest_point_index = None
        
        # Representing the state
        self.state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

        self.Q = np.eye(4)
        self.R = np.eye(1)

        # For visualisation
        self.canvas = {}
        self.batch = pyglet.graphics.Batch()


    def record_waypoints(self, cx, cy, cyaw, ck):
        #Initialise waypoints for planner
        self.cx=cx
        self.cy=cy
        self.cyaw = cyaw
        self.ck=ck
        self.old_nearest_point_index = None
        


    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi



    def calc_nearest_index(self):

        dx = [self.state.x - icx for icx in self.cx]
        dy = [self.state.y - icy for icy in self.cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = math.sqrt(mind)

        dxl = self.cx[ind] - self.state.x
        dyl = self.cy[ind] - self.state.y

        angle = self.pi_2_pi(self.cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def solve_dare(self, A, B):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        x = self.Q
        x_next = self.Q
        max_iter = 150
        eps = 0.01

        for i in range(max_iter):
            x_next = A.T @ x @ A - A.T @ x @ B @ \
                    la.inv(self.R + B.T @ x @ B) @ B.T @ x @ A + self.Q
            if (abs(x_next - x)).max() < eps:
                break
            x = x_next

        return x_next



    def dlqr(self, A, B):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + self.R) @ (B.T @ X @ A)

        eig_result = la.eig(A - B @ K)

        return K, X, eig_result[0]


    def lqr_steering_control(self, obs, pe, pth_e):
        
        self.state.x = obs['poses_x'][self.vehicleNumber]
        self.state.y = obs['poses_y'][self.vehicleNumber]
        self.state.yaw = obs['poses_theta'][self.vehicleNumber]
        self.state.v = obs['linear_vels_x'][self.vehicleNumber]
        
        ind, e = self.calc_nearest_index()

        k = self.ck[ind]
        v = self.state.v
        th_e = self.pi_2_pi(self.state.yaw - self.cyaw[ind])

        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        # print(A)

        B = np.zeros((4, 1))
        B[3, 0] = v / L

        K, _, _ = self.dlqr(A, B)

        x = np.zeros((4, 1))

        x[0, 0] = e
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt

        ff = math.atan2(L * k, 1)
        fb = self.pi_2_pi((-K @ x)[0, 0])

        delta = ff + fb

        return delta, ind, e, th_e