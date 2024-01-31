import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
import bisect
import sys
import cubic_spline_planner
import yaml
from PIL import Image, ImageOps, ImageDraw
import random
from datetime import datetime
import time
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba.experimental import jitclass
import pickle
import mapping
import cubic_spline_planner



def openConfigFile(configFileName):
    with open('configFiles/' + configFileName + '.yaml') as file:
        conf_dict = yaml.safe_load(file)
    conf_dict['name'] = configFileName
    conf = Namespace(**conf_dict)
    return conf


def load_config(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

def add_angles(a1, a2):
    angle = (a1+a2)%(2*np.pi)

    return angle

def sub_angles(a1, a2):
    angle = (a1-a2)%(2*np.pi)

    return angle

def add_angles_complex(a1, a2):
    real = math.cos(a1) * math.cos(a2) - math.sin(a1) * math.sin(a2)
    im = math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret



def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 

def transform_coords(x=[0, 0], theta=np.pi):
    # i want this function to transform coords from one coord system to another
    new_x = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    new_y = x[0] * np.sin(theta) + x[1] * np.cos(theta)

    return np.array([new_x, new_y])

def normalise_coords(x=[0, 0]):
    r = x[0]/x[1]
    y = np.sqrt(1/(1+r**2)) * abs(x[1]) / x[1] # carries the sign
    x = y * r
    return [x, y]

def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end


#@njit(cache=True)
def distance_between_points(x1, x2, y1, y2):
    distance = math.hypot(x2-x1, y2-y1)
    
    return distance

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def generate_circle_image():
    from matplotlib import image
    image = Image.new('RGBA', (600, 600))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 600), fill = 'black', outline ='black')
    draw.ellipse((50, 50, 550, 550), fill = 'white', outline ='white')
    draw.ellipse((150, 150, 450, 450), fill = 'black', outline ='black')
    draw.point((100, 100), 'red')
    image_path = sys.path[0] + '\\maps\\circle' + '.png'
    image.save(image_path, 'png')


def generate_circle_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '\\maps\\circle' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))

    R=10
    theta=np.linspace(0, 2*math.pi, 17)
    x = 15+R*np.cos(theta-math.pi/2)
    y = 15+R*np.sin(theta-math.pi/2)
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)
    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()
    return x, y, rx, ry, ryaw, rk, s


def generate_berlin_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '/maps/berlin' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    
    goals = [[16,3], [18,4], [18,7], [18,10], [18.5, 13], [19.5,16], [20.5,19], [19.5,22], [17.5,24.5], 
            [15.5,26], [13,26.5], [10,26], [7.5,25], [6,23], [7,21.5], [9.5,21.5], [11, 21.5], 
            [11,20], [10.5,18], [11,16], [12,14], [13,12], [13.5,10], [13.5,8], [14,6], [14.5,4.5], [16,3]]
    
    x = []
    y = []

    for xy in goals:
        x.append(xy[0])
        y.append(xy[1])
    
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)

    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()

    return x, y, rx, ry, ryaw, rk, s
    

def map_generator(map_name):
    map_config_path = sys.path[0] + '/maps/' + map_name + '.yaml'
    image_path = sys.path[0] + '/maps/' + map_name + '.png'
    with open(map_config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    map_conf = Namespace(**conf_dict)
    
    res=map_conf.resolution

    with Image.open(image_path) as im:
        gray_im = ImageOps.grayscale(im)
        map_array = np.asarray(gray_im)
        map_height = gray_im.height*res
        map_width = gray_im.width*res
        occupancy_grid = map_array<1
    
    return occupancy_grid, map_height, map_width, res 

   
def random_start(rx, ry, ryaw, distance_offset, angle_offset):
    
    #random.seed(datetime.now())
    
    '''
    if episode < 20000:
        if random.uniform(0,1)<0.1:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))
    
    elif episode >= 20000 and episode <50000:
        if random.uniform(0,1)<0.5:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))

    else:
    '''

    # i = int(random.uniform(0, len(x)-2))
    
    # #i = int(random.uniform(0, len(x)-2))
    # #i = int(random.uniform(10, 12))
    
    # next_i = (i+1)%len(y)
    # start_x = x[i] + (random.uniform(-distance_offset, distance_offset))
    # start_y = y[i] + (random.uniform(-distance_offset, distance_offset))
    
    # start_theta = math.atan2(y[next_i]-y[i], x[next_i]-x[i]) + (random.uniform(-angle_offset, angle_offset))
    # next_goal = (i+1)%len(x)

    i = int(random.uniform(0, len(rx)))
    start_x = rx[i] + random.uniform(-distance_offset, distance_offset)
    start_y = ry[i] + random.uniform(-distance_offset, distance_offset)
    start_theta = ryaw[i] + random.uniform(-angle_offset, angle_offset)
    next_i = 0

    return start_x, start_y, start_theta, next_i



def find_closest_point(rx, ry, x, y):

    dx = [x - irx for irx in rx]
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    
    ind = np.argmin(d)
    
    return ind

def check_closest_point(rx, ry, x, y, occupancy_grid, res, map_height):
    cp_ind = find_closest_point(rx, ry, x, y)   #find closest point 
    
    cp_x = rx[cp_ind]
    cp_y = ry[cp_ind]
    
    los_x = np.linspace(x, cp_x)
    los_y = np.linspace(y, cp_y)

    for x, y in zip(los_x, los_y):
        if occupied_cell(x, y, occupancy_grid, res, map_height):
            return True
    return False


def is_line_of_sight_clear(x1, y1, x2, y2, occupancy_grid, res, map_height):
    
    los_x = np.linspace(x1, x2)
    los_y = np.linspace(y1, y2)

    for x, y in zip(los_x, los_y):
        if occupied_cell(x, y, occupancy_grid, res, map_height):
            return False
    return True

def find_correct_closest_point(rx, ry, x, y, occupancy_grid, res, map_height):
    
    ind = find_closest_point(rx, ry, x, y)
    cpx = rx[ind]
    cpy = ry[ind]
    if is_line_of_sight_clear(x, y, cpx, cpy, occupancy_grid, res, map_height):
        return ind
    else:
        dx = [x - irx for irx in rx]
        dy = [y - iry for iry in ry]
        d = np.hypot(dx, dy)    
        inds = np.argsort(d)

        for i in inds:
            cpx = rx[i]
            cpy = ry[i]
            if is_line_of_sight_clear(x, y, cpx, cpy, occupancy_grid, res, map_height):
                return i
        else:
            print('No line of sight to centerline')
            return ind

def convert_xy_to_sn(rx, ry, ryaw, x, y, ds):
    dx = [x - irx for irx in rx]    
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    #Get distances from (x,y) to each point on centerline    
    ind = np.argmin(d)      #Index of s coordinate
    s = ind*ds              #Exact position of s
    n = d[ind]              #n distance (unsigned), not interpolated

    #Get sign of n by comparing angle between (x,y) and (s,0), and the angle of the centerline at s
    xy_angle = np.arctan2((y-ry[ind]),(x-rx[ind]))      #angle between (x,y) and (s,0)
    yaw_angle = ryaw[ind]                               #angle at s
    angle = sub_angles_complex(xy_angle, yaw_angle)     
    if angle >=0:   #Vehicle is above s line
        direct=1    #Positive n direction
    else:           #Vehicle is below s line
        direct=-1   #Negative n direction

    n = n*direct   #Include sign 

    return s, ind, n

def find_angle(A, B, C):
    # RETURNS THE ANGLE BÂC
    vec_AB = A - B
    vec_AC = A - C 
    dot = vec_AB.dot(vec_AC)
    #dot = (A[0] - C[0])*(A[0] - B[0]) + (A[1] - C[1])*(A[1] - B[1])
    magnitude_AB = np.linalg.norm(vec_AB)
    magnitude_AC = np.linalg.norm(vec_AC)

    angle = np.arccos(dot/(magnitude_AB*magnitude_AC))
    
    return angle

def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    print(np.degrees(angle))

    return angle



def convert_sn_to_xy(s, n, csp):
    x = []
    y = []
    yaw = []
    ds = []
    c = []
    
    for i in range(len(s)):
        ix, iy = csp.calc_position(s[i])
        if ix is None:
            break
        i_yaw = csp.calc_yaw(s[i])
        ni = n[i]
        fx = ix + ni * math.cos(i_yaw + math.pi / 2.0)
        fy = iy + ni * math.sin(i_yaw + math.pi / 2.0)
        x.append(fx)
        y.append(fy)

    # calc yaw and ds
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        yaw.append(math.atan2(dy, dx))
        ds.append(math.hypot(dx, dy))
        yaw.append(yaw[-1])
        ds.append(ds[-1])

    # calc curvature
    #for i in range(len(yaw) - 1):
    #    c.append((yaw[i + 1] - yaw[i]) / ds[i])
    c = 0

    return x, y, yaw, ds, c

def generate_line(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s, csp

def find_angle_to_line(ryaw, theta):

    angle = np.abs(sub_angles_complex(ryaw, theta))

    return angle


@njit(cache=True)
def occupied_cell(x, y, occupancy_grid, res, map_height):
    
    cell = (np.array([map_height-y, x])/res).astype(np.int64)

    if occupancy_grid[cell[0], cell[1]] == True:
        return True
    else:
        return False


spec = [('lidar_res', float32),
        ('n_beams', int32),
        ('max_range', float32),
        ('fov', float32),
        ('occupancy_grid', bool_[:,:]),
        ('map_res', float32),
        ('map_height', float32),
        ('beam_angles', float64[:])]


@jitclass(spec)
class lidar_scan():
    def __init__(self, lidar_res, n_beams, max_range, fov, occupancy_grid, map_res, map_height):
        
        self.lidar_res = lidar_res
        self.n_beams  = n_beams
        self.max_range = max_range
        self.fov = fov
        
        #self.beam_angles = (self.fov/(self.n_beams-1))*np.arange(self.n_beams)
        
        self.beam_angles = np.zeros(self.n_beams, dtype=np.float64)
        for n in range(self.n_beams):
            self.beam_angles[n] = (self.fov/(self.n_beams-1))*n

        self.occupancy_grid = occupancy_grid
        self.map_res = map_res
        self.map_height = map_height

    def get_scan(self, x, y, theta):
        
        scan = np.zeros((self.n_beams))
        coords = np.zeros((self.n_beams, 2))
        
        for n in range(self.n_beams):
            i=1
            occupied=False

            while i<(self.max_range/self.lidar_res) and occupied==False:
                x_beam = x + np.cos(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                y_beam = y + np.sin(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                occupied = occupied_cell(x_beam, y_beam, self.occupancy_grid, self.map_res, self.map_height)
                i+=1
            
            coords[n,:] = [np.round(x_beam,3), np.round(y_beam,3)]
            #dist = np.linalg.norm([x_beam-x, y_beam-y])
            dist = math.sqrt((x_beam-x)**2 + (y_beam-y)**2)
            
            scan[n] = np.round(dist,3)

        return scan, coords

def generate_initial_condition(name, episodes, distance_offset, angle_offset, vel_select):
    file_name = 'test_initial_condition/' + name
   
    initial_conditions = []
   
    track = mapping.map(name)
    track.find_centerline()
    goal_x = track.centerline[:,0]
    goal_y = track.centerline[:,1]
    rx, ry, ryaw, rk, d = cubic_spline_planner.calc_spline_course(goal_x, goal_y)
    
    k = [i for i in range(len(rk)) if abs(rk[i])>1]
    spawn_ind = np.full(len(rx), True)
    for i in k:
        spawn_ind[np.arange(i-10, i+5)] = False
    
    x = [rx[i] for i in range(len(rx)) if spawn_ind[i]==True]
    y = [ry[i] for i in range(len(ry)) if spawn_ind[i]==True]
    yaw = [ryaw[i] for i in range(len(ryaw)) if spawn_ind[i]==True]
    
    for eps in range(episodes):
        x_s, y_s, theta_s, current_goal = random_start(x, y, yaw, distance_offset, angle_offset)
        #x, y, theta = random_start(goal_x, goal_y, rx, ry, ryaw, rk, d, distance_offset, angle_offset)
        v_s = random.random()*(vel_select[1]-vel_select[0])+vel_select[0]
        delta_s = 0
        i = {'x':x_s, 'y':y_s, 'v':v_s, 'delta':delta_s, 'theta':theta_s, 'goal':current_goal}
        initial_conditions.append(i)

    #initial_conditions = [ [] for _ in range(episodes)]

    x = [initial_conditions[i]['x'] for i in range(len(initial_conditions))]
    y = [initial_conditions[i]['y'] for i in range(len(initial_conditions))]  
    
    plt.imshow(track.gray_im, extent=(0,track.map_width,0,track.map_height))
    plt.plot(rx, ry)
    plt.plot(goal_x, goal_y, 'o')
    plt.plot(x, y, 'x')
    plt.show()


    outfile=open(file_name, 'wb')
    pickle.dump(initial_conditions, outfile)
    outfile.close()


def plot_frenet_polynomial():

    
    ds = 0.1
    x_sparse = np.array([0,10])
    y_sparse = [0,0]
    rx, ry, ryaw, rk, s, csp = generate_line(x_sparse, y_sparse)
    
    x = 1
    y = 1
    #transform_XY_to_NS(rx, ry, x, y)
    convert_xy_to_sn(rx, ry, ryaw, x, y, ds)
    
    # s_0s = np.array([0, 0, 0])
    # n_0s = np.array([0.25, 0.25, 0.25])
    # thetas = np.array([0.5, 0.5, 0.5])
    # n_1s = np.array([0.7, 0.9, -0.9])

    s_0s = np.array([0,0])
    n_0s = np.array([0,0])
    thetas = np.array([0.8,0.8])
    n_1s = np.array([0.7,0.7])


    s_1s = s_0s+1
    s_2s = s_1s+0.5
    
    s_ = [[] for _ in range(len(s_0s))]
    n_ = [[] for _ in range(len(s_0s))]

    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    fig, ax = plt.subplots(1, figsize=(5.5,2.3))
    
    
    color='grey'
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

    # ax.set_xticks(ticks=[s_0s[0], s_1s[0]], labels=['$s_0$', '$s_1$'])
    # ax.set_yticks(ticks=[n_0s[0], n_1s[0]], labels=['$n_0$', '$n_1$'])



    ax.set_xticks(ticks=[], labels=[])
    ax.set_yticks(ticks=[], labels=[])

    # ax.plot(s_0s[0], n_0s[0], 'x', label='True position')
    # ax.plot([s_0s[1]], n_0s[1], 'x', label='Believed position')
    # ax.plot([0,s_2], [-1,-1])
    

    for idx, (s_0, n_0, theta, n_1, s_1, s_2) in enumerate(zip(s_0s, n_0s, thetas, n_1s, s_1s, s_2s)): 
       
        
        A = np.array([[3*s_1**2, 2*s_1, 1, 0], [3*s_0**2, 2*s_0, 1, 0], [s_0**3, s_0**2, s_0, 1], [s_1**3, s_1**2, s_1, 1]])
        B = np.array([0, theta, n_0, n_1])
        x = np.linalg.solve(A, B)
        #print(x)

        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]

        s = np.linspace(s_0, s_1)
        n = a*s**3 + b*s**2 + c*s + d
        s = np.concatenate((s, np.linspace(s_1, s_2)))
        n = np.concatenate((n, np.ones(len(np.linspace(s_1, s_2)))*n_1))
        
        s_[idx].append(s)
        n_[idx].append(n)
    
    alpha=0.7

    labels = ['Path using true position', 'Path using believed position']
    for i in range(len(s_0s)):
        if i==1:
            # plt.plot(s_[i][0], n_[i][0], label=labels[i], alpha=alpha)
            pass

    plt.plot(s_[1][0]-s_0s[1], n_[1][0]-n_0s[1], label='', alpha=0.5, color='red', linestyle='dashdot')

    # plt.plot(s_[0][0], n_[0][0], color='#1f77b4', label='Sampled path')
    # plt.fill_between(x=s_[1][0], y1=n_[1][0], y2=n_[2][0], color='#1f77b4', alpha=0.3, label='Range of selectable paths')


    # plt.plot(np.linspace(s_1, s_2), np.ones(len(np.linspace(s_1, s_2)))*n_1)
    # ax.hlines(y=1, xmin=-10, xmax=10, color='k', label='Track boundaries')
    # ax.hlines(y=-1, xmin=-10, xmax=10, color='k', label='_nolegend_')
    # ax.hlines(y=0, xmin=s_0-10, xmax=10, color='grey', linestyle='--', label='Track centerline')

    ax.grid(True)
    # ax.set_xlim([np.min(s_0s)-0.2, np.max(s_2s)+0.2])
    ax.set_xlabel('Distance along \ncenterline, $s$ [m]')
    ax.set_ylabel('Distance perpendicular \nto centerline, $n$ [m]')
    fig.tight_layout()


    # fig.subplots_adjust(right=0.5)
    # plt.figlegend(loc='center right', ncol=1)
    plt.show()



class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


if __name__ == '__main__':
    
    # plot_frenet_polynomial()

    # num = 1
    # noise = OUActionNoise(mu=np.zeros(num), sigma=0.03, theta=1, dt=0.01, x0=None)
    # y = np.zeros((600, num))
    # for i in range(len(y)):
    #     y[i,:] = noise()

    # for i in range(num):
    #     plt.plot(y[:,i])
    
    # plt.show()


    # def velocity_along_line(theta, velocity, ryaw, )
    
    #generate_berlin_goals()
    #x, y, rx, ry, ryaw, rk, s = generate_circle_goals()
    #start_x, start_y, start_theta, next_goal = random_start(x, y, rx, ry, ryaw, rk, s)

    #image_path = sys.path[0] + '/maps/' + 'circle' + '.png'       
    #occupancy_grid, map_height, map_width, res = map_generator(map_name='circle')
    #a = lidar_scan(res, 3, 10, np.pi, occupancy_grid, res, 30)
    #print(a.get_scan(15,5,0))
    
    # generate_initial_condition('porto_1', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('columbia_1', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('circle', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('berlin', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('torino', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('redbull_ring', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('f1_esp', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('f1_gbr', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('f1_mco', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])

    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    #plt.plot(start_x, start_y, 'x')
    #print(start_theta)
    #plt.arrow(start_x, start_y, math.cos(start_theta), math.sin(start_theta))
    #plt.plot(x, y, 's')
    #plt.plot(x[next_goal], y[next_goal], 'o')
    #plt.show()
    
    pass

