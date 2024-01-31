
from re import I
import sys
import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
import cubic_spline_planner
from PIL import Image, ImageOps, ImageDraw
import functions
from scipy import ndimage 
import pandas as pd 

import cubic_spline_planner



class map:
    def __init__(self, map_name):
        self.map_name = map_name
        self.read_yaml_file()
        self.load_map()
        self.occupancy_grid_generator()
    
    def read_yaml_file(self):
        file_name = sys.path[0] + '/maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            yaml_file = dict(yaml.full_load(file).items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.map_img_name = yaml_file['image']
    
    def load_map(self):
        image_path = sys.path[0] + '/maps/' + self.map_name + '.png'
        with Image.open(image_path) as im:
            self.gray_im = ImageOps.grayscale(im)
        self.map_height = self.gray_im.height*self.resolution
        self.map_width = self.gray_im.width*self.resolution
            
    def occupancy_grid_generator(self):  
        self.map_array = np.asarray(self.gray_im)
        self.occupancy_grid = self.map_array<1
        
    def find_centerline(self, show=False):
        
        self.dt = ndimage.distance_transform_edt(np.flipud(np.invert(self.occupancy_grid)))
        self.dt = np.array(self.dt *self.resolution)
        dt = self.dt

        d_search = 0.8
        # d_search = 0.5
        # d_search = 1
        n_search = 20
        dth = (np.pi * 4/5) / (n_search-1)

        # makes a list of search locations
        search_list = []
        for i in range(n_search):
            th = -np.pi/2 + dth * i
            x = -np.sin(th) * d_search
            y = np.cos(th) * d_search
            loc = [x, y]
            search_list.append(loc)

        pt = start = np.array([0, 0]) #TODO: start from map position
        self.cline = [pt]
        #th = self.stheta
        th = 0

        while (functions.get_distance(pt, start) > d_search/2 or len(self.cline) < 10) and len(self.cline) < 1000:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = functions.transform_coords(search_list[i], -th)
                search_loc = functions.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.xy_to_row_column(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = functions.transform_coords(search_list[ind], -th)
            pt = functions.add_locations(pt, d_loc)
            self.cline.append(pt)

            if show:
                self.plot_raceline_finding()

            th = functions.get_bearing(self.cline[-2], pt)
            #print(f"Adding pt: {pt}")

        self.cline = np.array(self.cline)
        self.N = len(self.cline)
        #print(f"Raceline found --> n: {len(self.cline)}")
        if show:
            self.plot_raceline_finding(True)
        #self.plot_raceline_finding(False)

        self.centerline = np.array(self.cline)
        self.centerline[:,0] = self.centerline[:,0]
        self.centerline[:,1] = self.centerline[:,1]
        
        self.centerline_1 = self.centerline[1:-1,:].copy()
        self.centerline_1 = self.centerline_1[:][np.arange(0,len(self.centerline_1[:]),2)]
        self.centerline_1 = np.append(self.centerline_1, self.centerline_1[0]+np.array([-0.01,0]))

        self.centerline_1 = np.reshape(self.centerline_1, (int(len(self.centerline_1)/2), 2) )
        self.centerline = self.centerline_1.copy()

        self.rx, self.ry, self.ryaw, self.rk, self.d = cubic_spline_planner.calc_spline_course(self.centerline[:,0], self.centerline[:,1])
        
    

    def plot_raceline_finding(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.dt, origin='lower')

        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, '+', markersize=16)

        for pt in self.search_space:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, 'x', markersize=12)


        plt.pause(0.001)

        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.dt.shape[1]:
            c = self.dt.shape[1] - 1
        if r >= self.dt.shape[0]:
            r = self.dt.shape[0] - 1

        return c, r
    




def test_map():
    m = map('f1_esp')
    m.find_centerline(False)
    plt.imshow(m.gray_im, extent=(m.origin[0], m.origin[0] + m.map_width, m.origin[1], m.origin[1]+m.map_height))
    plt.plot(m.rx,m.ry)
    plt.show()


def generate_csv():
    mapName = 'f1_esp'
    
    m = map(mapName)
    m.find_centerline(True)
    
    centerlineData = {
        'x': m.rx,
        'y': m.ry,
        'yaw': m.ryaw,
        'curve': m.rk,
        'distance': m.d
    }
    
    centerlineDataframe = pd.DataFrame(centerlineData)
    centerlineDataframe.to_csv(sys.path[0] + '/maps/' + mapName + '_centerline.csv')


if __name__=='__main__':
    # test_map()
    generate_csv()
