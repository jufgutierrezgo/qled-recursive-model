from cgi import print_arguments
from cmath import cos
from email.policy import default
from functools import partial

import numpy as np 
from numpy import loadtxt

import sys
np.set_printoptions(threshold=sys.maxsize)

import array
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


# annotating a variable with a type-hint
from typing import List, Tuple


from scipy.fftpack import diff


from scipy import stats
from scipy import signal

# import the required library
import torch

import luxpy as lx

import timeit

import os

#Class constants
class Constants:
    # global variables

    #Array with normal vectors for each wall.
    NORMAL_VECTOR_WALL = [[0,0,-1],[0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1]]
    #directory root of the project
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    #directory to save channel impulse response raw data
    SENSOR_PATH = ROOT_DIR + "/sensors/"
    #directory to save histograms and graphs  
    #REPORT_PATH = ROOT_DIR + "/report/"
    #Numbers of LED (Transmission channels)
    NO_LEDS = 4
    #Numbers of Photodetector Channels
    NO_DETECTORS = 3

#Class for the TRansmitter
class Transmitter:    

    # The init method or constructor
    def __init__(self, name):
           
        # Instance Variable
        self.name = name
    
    # Set the [x y z] position vector.
    def set_position(self, position):
        self.position = np.array(position)

    # Set the [x y z] normal vector of the transmitter.
    def set_normal(self, normal):
        self.normal = np.array([normal])    
    
    # Set the scalar lambert number.
    def set_mlambert(self, mlambert):
        self.mlambert = mlambert

    # Set the scalar power of the LED in Watts.
    def set_power(self, power):
        self.power = power 

    # Set the [l1 l2 l3 l4] vector of the central wavelengths in [nm].
    def set_wavelengths(self, wavelengths):
        self.wavelengths = np.array(wavelengths)
    
    # Set the [d1 d2 d3 d4] vector of the full width at half maximum of each central wavelengths, in [nm].
    def set_fwhm(self, fwhm):
        self.fwhm = np.array(fwhm)

    # Print set of parameters.
    def get_parameters(self):
        print('\nList of parameters for LED transmitter:')
        print('Position [x y z]: ', self.position)
        print('Normal Vector [x y z]: ', self.normal)
        print('Lambert Number: ', self.mlambert)
        print('Power[W]: ', self.power)
        print('Central Wavelengths[nm]: ', self.wavelengths)
        print('FWHM[nm]: ', self.fwhm)

    def led_pattern(self) -> None:
        """Function to create a 3d radiation pattern of the LED source.
        
        The LED for recurse channel model is assumed as lambertian radiator. The number of lambert 
        defines the directivity of the light source.
            
        Parameters:
            m: Lambert number
        
        Returns: None.

        """

        theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0,np.pi/2, 40)
        THETA, PHI = np.meshgrid(theta, phi)
        R = (self.mlambert +1)/(2*np.pi)*np.cos(PHI)**self.mlambert
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        plot = ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)

        plt.show()
        return 0

#Class for the photodetector
class Photodetector:


    #Class constants

    # The init method or constructor
    def __init__(self, name):
           
        # Instance Variable
        self.name = name
    
    # Set the [x y z] position vector.
    def set_position(self, position):
        self.position = np.array(position)

    # Set the [x y z] normal vector.
    def set_normal(self, normal):
        self.normal = np.array([normal])

    # Set active area in [m2].
    def set_area(self, area):
        self.area = np.array(area)    

    
    # Set FOV of the detector.
    def set_fov(self, fov):
        self.fov = fov
    
    # Set the type of responsivity profile
    def set_responsivity(self, sensor):
        self.sensor = sensor 

        if self.sensor == 'TCS3103-04':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityTCS3103-04.txt")                       
        elif self.sensor == 'S10917-35GT':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityS10917-35GT.txt")                       
        else:
            print("Sensor reference not valid.")  

    # Plot the spectral responsivity of the photodetector.
    def plot_responsivity(self):
        plt.plot(self.responsivity[:,0],self.responsivity[:,1],color='r', linestyle='dashed') 
        plt.plot(self.responsivity[:,0],self.responsivity[:,2],color='g', linestyle='dashed') 
        plt.plot(self.responsivity[:,0],self.responsivity[:,3],color='b', linestyle='dashed')
        plt.title("Spectral Responsiity of Photodetector")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Responsivity [A/W]")
        plt.grid()
        plt.show()

    # Print set of parameters.
    def get_parameters(self):
        print('\nList of parameters for photodetector:')
        print('Position [x y z]: ', self.position)
        print('Normal Vector [x y z]: ', self.normal)
        print('Active Area[m2]: ', self.area)
        print('FOV: ', self.fov)        
        print('Responsivity: ', self.sensor)

#Class for the environment
class Indoorenvironment:        

    # The init method or constructor
    def __init__(self, name):
           
        # Instance Variable
        self.name = name    
    
    # Set the [x y z] position vector.
    def set_size(self, size):
        self.size = np.array(size)

    # Set scalar order reflection 
    def set_noreflections(self, no_reflections):
        self.no_reflections = no_reflections    
    
    # Set the vector of reflectance at central wavelengths.
    def set_reflectance(self, wall_name, reflectance_wall):
        self.wall_name = wall_name
        self.reflectance_wall = np.array(reflectance_wall)

        if self.wall_name == 'ceiling':
            self.ceiling = self.reflectance_wall
        elif self.wall_name == 'west':
            self.west = self.reflectance_wall
        elif self.wall_name == 'north':
            self.north = self.reflectance_wall
        elif self.wall_name == 'east':
            self.east = self.reflectance_wall
        elif self.wall_name == 'south':
            self.south = self.reflectance_wall
        elif self.wall_name == 'floor':
            self.floor = self.reflectance_wall
        else: 
            print('Invalid wall name.')

    # Set distance between points in cm.
    def set_pointresolution(self, resolution):
        self.resolution = resolution

    # Print set of parameters.
    def get_parameters(self):
        print('\nList of parameters for indoor envirionment:')
        print('Size [x y z] -> [m]: ', self.size)
        print('Order reflection: ', self.no_reflections)
        print('Reflectance at central wavelengths [r1 r2 r3 r4]: ', self.reflectance)
        print('Resolution points [cm]: ', self.resolution)        
        
    # Create 3D coordinates of all points in the model
    def create_grid(self,tx_position,rx_position):                

        #Number of ticks in each axis, based on spatial resolution. 
        no_xtick = int(self.size[0]/self.resolution)
        no_ytick = int(self.size[1]/self.resolution)
        no_ztick = int(self.size[2]/self.resolution)

        print('\nGrid Parameters:')
        print("Number of ticks [x y z]:",no_xtick,no_ytick,no_ztick)

        #Creates arrays for save a points in every wall
        ceiling_points = np.zeros((no_xtick*no_ytick,3),dtype=np.float16)
        west_points = np.zeros((no_ztick*no_xtick,3),dtype=np.float16)
        north_points = np.zeros((no_ztick*no_ytick,3),dtype=np.float16)
        east_points = np.zeros((no_ztick*no_xtick,3),dtype=np.float16)
        south_points = np.zeros((no_ztick*no_ytick,3),dtype=np.float16)
        floor_points = np.zeros((no_xtick*no_ytick,3),dtype=np.float16)

        #Creates normal vector for each point
        ceiling_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[0]],no_xtick*no_ytick,axis=0)
        east_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[1]],no_ztick*no_xtick,axis=0)
        south_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[2]],no_ztick*no_ytick,axis=0)
        west_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[3]],no_ztick*no_xtick,axis=0)
        north_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[4]],no_ztick*no_ytick,axis=0)
        floor_normal = np.repeat([Constants.NORMAL_VECTOR_WALL[5]],no_xtick*no_ytick,axis=0)
        
        #Creates reflectance vector for each point
        ceiling_reflectance = np.repeat([self.ceiling],no_xtick*no_ytick,axis=0)
        west_reflectance = np.repeat([self.west],no_ztick*no_xtick,axis=0)
        north_reflectance = np.repeat([self.north],no_ztick*no_ytick,axis=0)
        east_reflectance = np.repeat([self.east],no_ztick*no_xtick,axis=0)
        south_reflectance = np.repeat([self.south],no_ztick*no_ytick,axis=0)
        floor_reflectance = np.repeat([self.floor],no_xtick*no_ytick,axis=0)
               
        #Array with ticks coordinates in every axis
        x_ticks = np.linspace(self.resolution/2,self.size[0]-self.resolution/2,no_xtick)
        y_ticks = np.linspace(self.resolution/2,self.size[1]-self.resolution/2,no_ytick)
        z_ticks = np.linspace(self.resolution/2,self.size[2]-self.resolution/2,no_ztick)

        #Computes the total number of points. If the door is not included, the rx position point is added at end of the array points                
        self.no_points=2*no_xtick*no_ytick + 2*no_ztick*no_xtick + 2*no_ztick*no_ytick   + 2 

        #Generates the x,y,z of grids in each points
        x_ygrid,y_xgrid = np.meshgrid(x_ticks,y_ticks)
        x_zgrid,z_xgrid = np.meshgrid(x_ticks,z_ticks)
        y_zgrid,z_ygrid = np.meshgrid(y_ticks,z_ticks)

        #Save x,y,z coordinates of points in each wall
        ceiling_points[:,0] = floor_points[:,0] = x_ygrid.flatten() 
        ceiling_points[:,1] = floor_points[:,1] = y_xgrid.flatten() 
        ceiling_points[:,2] , floor_points[:,2] = self.size[2] , 0        
        
        west_points[:,0] = east_points[:,0] = x_zgrid.flatten() 
        west_points[:,2] = east_points[:,2] = z_xgrid.flatten() 
        east_points[:,1] , west_points[:,1] = 0 , self.size[1]
        
        north_points[:,1] = south_points[:,1] = y_zgrid.flatten() 
        north_points[:,2] = south_points[:,2] = z_ygrid.flatten() 
        south_points[:,0] , north_points[:,0] = 0 , self.size[0]    
        
        
        #Creates tensors for gridpoints, normal vectors and reflectance vectors.        
        self.gridpoints = torch.from_numpy(np.concatenate((ceiling_points,east_points,south_points,west_points,north_points,floor_points,[tx_position],[rx_position]),axis=0))          
        #self.normal_vectors = torch.from_numpy(np.concatenate((ceiling_normal,east_normal,south_normal,west_normal,north_normal,floor_normal,[Constants.NORMAL_VECTOR_WALL[0]],[Constants.NORMAL_VECTOR_WALL[5]]),axis=0,dtype=np.int8)).reshape(self.no_points,1,3)       
        self.normal_vectors = torch.from_numpy(np.concatenate((ceiling_normal,east_normal,south_normal,west_normal,north_normal,floor_normal,led1.normal,pd1.normal),axis=0,dtype=np.float16)).reshape(self.no_points,1,3)       
        self.reflectance_vectors = np.concatenate((ceiling_reflectance,east_reflectance,south_reflectance,west_reflectance,north_reflectance,floor_reflectance,[[0,0,0,0]],[[0,0,0,0]]),axis=0,dtype=np.float16)               
         
        #print("Grid Points->",self.gridpoints)
        

        #Delta area calculation
        self.deltaA = (2*self.size[0]*self.size[1] + 2*self.size[0]*self.size[2] + 2*self.size[1]*self.size[2])/(self.no_points-2)
        
        print("The total number of points is: ",self.no_points)
        print("DeltaA: ",self.deltaA)
        print("//-------- points array created --------------//")


        return 0       

    def create_parameters(self,fov):
        """This function creates an 3d-array with cross-parametes between points. 
        
        This parameters are the distance between points and the cosine of the angles 
        respect to the normal vector. Using this array is commputed the channel immpulse 
        response.
        
        Parameters:
            gridpoints: 2d tensor array with [x,y,z] coordinates for each point. 
            normal_vector: 2d tensor array with [x,y,z] coordinates of normal vector in each point            

        Returns: Returns a 3d-array with distance and cos(tetha) parameters. The 
        shape of this array is [2,no_points,no_points].
        
        
            _____________________    
           /                    /|
          /                    / |
         /                    /  |
        /____________________/  /| 
        |     Distance       | / |
        |____________________|/ /
        |     Cos(tetha)     | /
        |____________________|/
        

        """
    
        #Numpy array 3D to save paiswise distance and cos_phi. 
        self.wall_parameters = np.zeros((2,self.no_points,self.no_points),dtype=np.float16)         

        
        #Computes pairwise-element distance using tensor
        dist = torch.cdist(self.gridpoints,self.gridpoints)                
        print("Distance shape->",dist.shape)
        #print("Distance ->",dist)
        
        #Computes the pairwise-difference (vector) using tensor
        diff = -self.gridpoints.unsqueeze(1) + self.gridpoints
        print("Difference shape->",diff.shape)        
        #print("Difference ->",diff)        
        
        #Computes the unit vector from pairwise-difference usiing tensor
        unit_vector = torch.nan_to_num(torch.div( diff ,dist.reshape(self.no_points,self.no_points,1)),nan=0.0)
        print("Unitec vector shape ->",unit_vector.shape)

        #Computes the cosine of angle between unit vector and normal vector using tensor.
        cos_phi = torch.sum(unit_vector*self.normal_vectors,dim=2)        
        print("Cosine shape->",cos_phi.shape)
        #print("Cosine->",cos_phi[-1,:])

        array_rx = np.asarray(cos_phi[-1,:])
        low_values_flags = array_rx < np.cos(fov*np.pi/180)  # Where values are low
        array_rx[low_values_flags] = 0  # All low values set to 0
        #print("FOV->",array_np)
        #print("FOV->",np.cos(fov*np.pi/180))  

        array_tx = np.asarray(cos_phi[-2,:])
        low_values_flags = array_tx < np.cos(90*np.pi/180)  # Where values are low
        array_tx[low_values_flags] = 0  # All low values set to 0
        
        #print("Cosine->",cos_phi[-1,:])
        

        #Save in numpy array the results of tensor calculations
        self.wall_parameters[0,:,:] = dist.numpy()
        self.wall_parameters[1,:,:] = cos_phi.numpy()

        print("//------- parameters array created -----------//")
        #np.set_printoptions(threshold=np.inf)        
        #numpy.savetxt("ew_par_dis.csv", ew_par[0,:,:], delimiter=",")  
        #numpy.savetxt("ew_par_cos.csv", ew_par[1,:,:], delimiter=",")  

        return 0

#Class for Recursive Model computations
class Recursivemodel:
    """ This class contains the function to calculates the CIR and DC-gain in the optical channel. """

    # The init method or constructor
    def __init__(self, name,led,photodetector,room):
           
        # Instance Variable
        self.name = name
        self.led = led
        self.photodector = photodetector
        self.room = room
    
    #Function to compute the CIR
    def compute_cir(self):        
        """ Function to compute the channel impulse response for each reflection. 
    
        Parameters:
            led.m: lambertian number to tx emission                         
            led.wall_parameters: 3D array with distance and cosine pairwise-elemets.              
            pd.area: sensitive area in photodetector
            

        Returns: A list with 2d-array [power_ray,time_delay] collection for each 
        refletion [h_0,h_1,...,h_k].
        

        """       
        
        #defing variables and arrays
        tx_index_point = room.no_points-2                
        rx_index_point = room.no_points-1                
        
        cos_phi = np.zeros((room.no_points),dtype=np.float16)
        dis2 = np.zeros((room.no_points,room.no_points),dtype=np.float16)

        h0_se = np.zeros((room.no_points,4),dtype=np.float64)
        h0_er = np.zeros((room.no_points,1),dtype=np.float64)                  
                
       
        #Time delay between source and each cells 
        #h0_se[:,1] = room.wall_parameters[0,tx_index_point,:]/SPEED_OF_LIGHT
        #Time delay between receiver and each cells 
        #h0_er[:,1] = room.wall_parameters[0,rx_index_point,:]/SPEED_OF_LIGHT

        #define distance^2 and cos_phi arrays
        dis2 = np.power(room.wall_parameters[0,:,:],2)            
        cos_phi = room.wall_parameters[1,int(tx_index_point),:]
        #print("COS_PHI for Tx->",cos_phi)
        

        tx_power = (led1.mlambert+1)/(2*np.pi)*np.multiply(np.divide(1,dis2[tx_index_point,:],out=np.zeros((room.no_points)), where=dis2[tx_index_point,:]!=0),np.power(cos_phi,led1.mlambert))
        rx_wall_factor = pd1.area*room.wall_parameters[1,int(rx_index_point),:]

        #Differential power between all grid points without reflectance
        dP_ij = np.zeros((room.no_points,room.no_points),np.float32)
        dP_ij = np.divide(room.deltaA*room.wall_parameters[1,:,:]*np.transpose(room.wall_parameters[1,:,:]),np.pi*dis2,out=np.zeros_like(dP_ij),where=dis2!=0)         
        #print("Differential Power of Points->",dP_ij)
        
        
        #Array creation for dc_gain and previuos dc_gain
        self.h_k = []
        hlast_er = []
        

        for i in range(room.no_reflections+1):
            
            #Creates the array to save h_k reflections response and last h_er response
            self.h_k.append(np.zeros((room.no_points,4),np.float64))
            hlast_er.append(np.zeros((room.no_points,4),np.float64)) 

            if i == 0:           
                
                #Magnitude of CIR in LoS
                self.h_k[i][0,:] = tx_power[int(rx_index_point)]*rx_wall_factor[int(tx_index_point)]
                
                #Time Delay of CIR in LoS
                #h_k[i][0,1] = room.wall_parameters[0,int(tx_index_point),int(rx_index_point)]/SPEED_OF_LIGHT

                print("|>>--------------h{}-computed--------------<<|".format(i))              
                #numpy.savetxt(CIR_PATH+"h0.csv", h_k[i], delimiter=",")
                #print(self.h_k[i])

            elif i==1:

                
                #hlast_er[i] = np.multiply(np.reshape(h0_er[:,0],(-1,1)),room.reflectance_vectors)               
                
                #Impulse response between source and each cells without reflectance. The reflectance is added in the h_k computing.
                h0_se = np.multiply(np.reshape(np.multiply(room.deltaA*tx_power,room.wall_parameters[1,:,int(tx_index_point)]),(-1,1)),room.reflectance_vectors)

                #Impulse response between receiver and each cells 
                h0_er[:,0] = np.divide(np.multiply(room.wall_parameters[1,:,int(rx_index_point)],rx_wall_factor),np.pi*dis2[rx_index_point,:],out=np.zeros((room.no_points)), where=dis2[rx_index_point,:]!=0)
                
                #print("h0_se array->:",h0_se[:,0])
                #print("h0_er array->:",h0_er[:,0])       
                
                #Previous h_er RGBY vectors of magnitude for LoS
                hlast_er[i] = np.repeat(h0_er,repeats=4,axis=1)

                #Current vector for h1 impulse response for RGBY
                #Red-Green-Blue-Yellow                
                self.h_k[i] = np.multiply(h0_se,hlast_er[i])

                #print("H1->", self.h_k[i][:,0])               

                #Time delay for h1 impulse response
                #h_k[i][:,4] = h0_se[:,1] + h0_er[:,1]
                

                print("|>>--------------h{}-computed--------------<<|".format(i))              
                #np.savetxt(CIR_PATH+"h1.csv", h_k[i], delimiter=","              
                

            elif i>=2:                

                #Computes the last h_er to compute h_k  
                #hlast_er[i] = np.multiply(hlast_er[i-1],np.multiply(room.reflectance_vectors,np.reshape(np.sum(dP_ij,axis=0),(-1,1))))
                for color in range(Constants.NO_LEDS):

                    hlast_er[i][:,color] = np.sum(np.multiply(hlast_er[i-1][:,color],np.multiply(room.reflectance_vectors[:,color],dP_ij)),axis=1)
                    print("hlast->",np.shape(hlast_er[i]))

                    self.h_k[i][:,color] = np.multiply(h0_se[:,0],hlast_er[i][:,color])
                    print("h_k->",np.shape(self.h_k[i]))    
                

                

                

                #Computes the current h_er 
                #hnext_er = np.multiply(hlast_er[i-1],np.dot(dP_ij,room.reflectance_vectors))

                #Computes the current h_k (h_se + h_er)
                #self.h_k[i] = np.multiply(np.reshape(h0_se[:,0],(-1,1)),hlast_er[i])                
                #partial_matrix = np.multiply(h0_se[:,0],np.multiply(np.multiply(np.reshape(hlast_er[i-1],(1,-1)),dP_ij)))                
                #self.h_k[i] = np.multiply(room.reflectance_vectors**2,np.reshape(np.sum(partial_matrix,axis=1),(-1,1)))
                #print(np.shape(room.reflectance_vectors[:,0]))
                #print(np.shape(np.multiply(hlast_er[i-1],room.reflectance_vectors[:,0]**2)))
                #h_er = np.multiply(np.reshape(np.multiply(hlast_er[i-1],np.reshape(room.reflectance_vectors[:,0],(-1,1))),(1,-1)),dP_ij)
                #print("h_er->",np.shape(h_er))
                #self.h_k[i] = np.reshape(np.sum(np.multiply(np.reshape(np.multiply(h0_se[:,0],room.reflectance_vectors[:,0]),(1,-1)),h_er),axis=1),(-1,1))
                #print("h_er->",np.shape(self.h_k[i]))
                #print("H2->", self.h_k[i][:,0])               

                print("|>>--------------h{}-computed--------------<<|".format(i))              
            
        return 0

    #This function calculates the total power received from LoS and h_k reflections
    def compute_dcgain(self):

        print("\nResults DC Gain [R G B Y]:")          
        self.h_dcgain = np.zeros((room.no_reflections+1,4),np.float32)

        for i in range(0,room.no_reflections+1):
            self.h_dcgain[i,:] = np.sum(self.h_k[i][0:-2,0:4], axis = 0) 
            print(" H"+str(i)+" RGBY DC Gain Power [W]:")          
            print(self.h_dcgain[i,:])


        self.rgby_dcgain = np.sum(self.h_dcgain, axis = 0)
        print("Total RGBY DC Gain Power [W]")
        print(self.rgby_dcgain)

        return 0

    # This function creates a gaussian function 
    def gaussian(x, mu, sig, amp):
        return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    #This function creates a SPD of LED from central wavelengths, FWHM and DC gain of channel
    def create_spd(self):        
        
        #Array for wavelenght points from 380nm to (782-2)nm with 2nm steps
        self.wavelenght = np.arange(380, 782, 2) 
        
        #Arrays to estimate the RGBY gain spectrum
        self.r_data = self.rgby_dcgain[0]*stats.norm.pdf(self.wavelenght, led1.wavelengths[0], led1.fwhm[0])
        self.g_data = self.rgby_dcgain[1]*stats.norm.pdf(self.wavelenght, led1.wavelengths[1], led1.fwhm[1])
        self.b_data = self.rgby_dcgain[2]*stats.norm.pdf(self.wavelenght, led1.wavelengths[2], led1.fwhm[2])
        self.y_data = self.rgby_dcgain[3]*stats.norm.pdf(self.wavelenght, led1.wavelengths[3], led1.fwhm[3])
        self.spd_data = [self.r_data , self.g_data , self.b_data , self.y_data]

        return 0

    #This function plots the SPD of QLED
    def plot_spd(self):

        ## plot red spd data
        plt.plot(self.wavelenght,self.r_data,'r')
        plt.plot(self.wavelenght,self.g_data,'g')
        plt.plot(self.wavelenght,self.b_data,'b')
        plt.plot(self.wavelenght,self.y_data,'y')
        plt.title("Spectral Power distribution of QLED")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Radiometric Power [W]")
        plt.grid()
        plt.show()

        return("SPD plotted.")

    #This function calculates a CCT and CRI of the QLED SPD.
    def compute_cct_cri(self):
        
        #Computing the xyz coordinates from SPD-RGBY estimated spectrum
        xyz = lx.spd_to_xyz([self.wavelenght,self.r_data + self.g_data + self.b_data + self.y_data])
        #Computing the CRI coordinates from SPD-RGBY estimated spectrum
        self.cri = lx.cri.spd_to_cri(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]))
        #Computing the CCT coordinates from SPD-RGBY estimated spectrum
        self.cct = lx.xyz_to_cct_ohno2014(xyz)
        #Print color data
        print("CCT: ", self.cct)
        print("CRI: ", self.cri)
        
        return ("CRI and CCT computed.")

    #This function calculates the irradiance.
    def compute_irradiance(self):        
        
        self.irradiance = lx.spd_to_power(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]),ptype = 'ru') 
        print("Irradiance [W/m2]: ", self.irradiance)

        return ("The irradiance on the dectector was computed.")

    #This function calculates the illuminance.
    def compute_illuminance(self):
        
        self.illuminance = lx.spd_to_power(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]),ptype = 'pu') 
        print("Illuminance [lx]: ", self.illuminance)

        return ("The illuminance on the dectector was computed.")

    #This function computes channel matrix
    def compute_channelmatrix(self):

        #Numpy array 2D to save the channel matrix
        self.channelmatrix = np.zeros((Constants.NO_DETECTORS,Constants.NO_LEDS),dtype=np.float32)         
        
        #print(self.wavelenght)
        #print(self.responsivity[:,0])

        for j in range(0,Constants.NO_LEDS):
            for i in range(1,Constants.NO_DETECTORS+1):
                #print(self.spd_data[j])
                #print(self.responsivity[:,i])
                #print(i,'-',j)
                self.channelmatrix[i-1][j] = np.dot(self.spd_data[j],pd1.responsivity[:,i])

        #Calculation of spectral interference. 
        print("Gain-Interference Matrix:")
        print(self.channelmatrix)        
        #print("{0:.15f}".format(self.channelmatrix[0][0]))
        #print("{0:.15f}".format(self.channelmatrix[2][1]))
        #print("{0:.15f}".format(self.channelmatrix[0][3]))
                
        return 0

starttime = timeit.default_timer()

#code to simulate a VLC channel

led1 = Transmitter("Led1")
led1.set_position([3.75,2.75,1])
led1.set_normal([0,0,1]) 
led1.set_mlambert(1)
led1.set_power(1)
led1.set_wavelengths([650,530,430,580])
led1.set_fwhm([20,12,20,20])
led1.get_parameters()
#led1.led_pattern()

pd1 =  Photodetector("PD1")
pd1.set_position([6,0.8,0.8])
pd1.set_normal([0,0,1])
pd1.set_area(1e-4)
pd1.set_fov(70)
pd1.set_responsivity('S10917-35GT')
pd1.plot_responsivity()
pd1.get_parameters()

room = Indoorenvironment("Room")
room.set_size([7.5,5.5,3.5])
room.set_noreflections(3)
room.set_pointresolution(1/6)
room.set_reflectance('ceiling',[0.69,0.69,0.69,0.69])
room.set_reflectance('west',[0.12,0.12,0.12,0.12])
room.set_reflectance('north',[0.58,0.58,0.58,0.58])
room.set_reflectance('east',[0.3,0.3,0.3,0.3])
room.set_reflectance('south',[0.56,0.56,0.56,0.56])
room.set_reflectance('floor',[0.09,0.09,0.09,0.09])    
room.create_grid(led1.position,pd1.position)
room.create_parameters(pd1.fov)

channel_model = Recursivemodel("ChannelModelB",led1,pd1,room)
channel_model.compute_cir()
channel_model.compute_dcgain()
channel_model.create_spd()
channel_model.plot_spd()
channel_model.compute_cct_cri()
channel_model.compute_irradiance()
channel_model.compute_illuminance()
channel_model.compute_channelmatrix()


#ending code


print("\nThe execution time is :", timeit.default_timer() - starttime)
print("Simulation finished.")