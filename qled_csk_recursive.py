from cgi import print_arguments
from cmath import cos
from email.policy import default
from functools import partial
from multiprocessing.dummy import Array
from tokenize import String
from turtle import position

import numpy as np 
from numpy import loadtxt
from numpy.core.function_base import linspace

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
    REPORT_PATH = ROOT_DIR + "/report/"
    #Numbers of LED (Transmission channels)
    NO_LEDS = 4
    #Numbers of Photodetector Channels
    NO_DETECTORS = 3
    #Speed of light
    SPEED_OF_LIGHT = 299792458

#Class for the TRansmitter
class Transmitter:    

    # The init method or constructor
    def __init__(self, 
        name: str, 
        position: Tuple[float,float,float], 
        normal: Tuple[float,float,float], 
        wavelengths: Tuple, 
        fwhm: Tuple, 
        mlambert: float =1, 
        power: float =1) -> None:
           
        # Instance Variable
        self._name = name
        self._position = np.array(position)
        self._normal = np.array([normal])  
        self._mlambert = mlambert
        self._power = power 
        self._wavelengths = np.array(wavelengths)
        self._fwhm = np.array(fwhm)

    #Name Property    
    @property
    def name(self) -> str:
        """The name property"""        
        return self._name

    @name.setter
    def name(self,value):        
        self._name =  value
    
    #Position Property    
    @property
    def position(self) -> Tuple[float,float,float]:
        """The position property"""    
        return self._position

    @position.setter
    def position(self,position):        
        self._position =  position    

    #Normal Property
    @property
    def normal(self) -> Tuple[float,float,float]:
        """The normal property"""        
        return self._normal

    @normal.setter
    def position(self,normal):        
        self._normal = np.array(normal)        
    
    #mLambert Property
    @property
    def mlambert(self)  -> float:
        """The Lambert number property"""        
        return self._mlambert

    @mlambert.setter
    def mlambert(self,mlabert):        
        self._mlambert =  mlabert

    #Power Property
    @property
    def power(self) -> float:
        """The Power property"""
        return self._power

    @power.setter
    def power(self,power):        
        self._power =  power


    #Wavelengths Property
    @property
    def wavelengths(self)  -> Tuple:
        """The Wavelengths property"""
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        self._wavelengths = np.array(wavelengths)

    #FWHM Property
    @property
    def fwhm(self) -> Tuple:
        """The FWHM property"""        
        return self._power

    @fwhm.setter
    def fwhm(self, fwhm):        
        self._fwhm = np.array(fwhm)

    def __str__(self) -> str:
        return (
            f'List of parameters for LED transmitter: \n'
            f'Position [x y z]: {self._position} \n'
            f'Normal Vector [x y z]: {self._normal} \n' 
            f'Lambert Number: {self._mlambert} \n'
            f'Power[W]: {self._power} \n'
            f'Central Wavelengths[nm]: {self._wavelengths} \n'
            f'FWHM[nm]: {self._fwhm}'
        )

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
        R = (self._mlambert +1)/(2*np.pi)*np.cos(PHI)**self._mlambert
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        plot = ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)

        plt.show()

#Class for the photodetector
class Photodetector:    

    # The init method or constructor
    def __init__(self, 
        name: str, 
        position: Tuple[float,float,float], 
        normal: Tuple[float,float,float], 
        area: Tuple[float,float,float], 
        sensor: str ="", 
        fov: float =90) -> None:
           
        # Instance Variable
        self._name = name
        self._position = np.array(position)
        self._normal = np.array([normal])
        self._area = np.array(area)    
        self._fov = fov
        self._sensor = sensor 
    
        if self.sensor == 'TCS3103-04':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityTCS3103-04.txt")
            print("Responsivity loaded succesfully")                       
        elif self.sensor == 'S10917-35GT':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityS10917-35GT.txt")                       
            print("Responsivity loaded succesfully")
        elif self.sensor == '': 
            print("Specify sensor reference")
        else:
            print("Sensor reference not valid")  

    #Name Property    
    @property
    def name(self) -> str:
        """The name property"""    
        return self._name

    @name.setter
    def name(self,value):        
        self._name =  value

    #Position Property    
    @property
    def position(self) -> Tuple[float,float,float]:
        """The position property"""
        return self._position

    @position.setter
    def position(self,position):        
        self._position =  position    

    #Normal Property
    @property
    def normal(self) -> Tuple[float,float,float]:
        """The normal property"""        
        return self._normal

    @normal.setter
    def position(self,normal):        
        self._normal = np.array(normal)        

    #Area Property    
    @property
    def area(self) -> float:
        """The position property"""        
        return self._area

    @area.setter
    def area(self,area):        
        self._area =  area

    #FOV Property    
    @property
    def fov(self)  -> float:
        """The position property"""
        return self._fov

    @fov.setter
    def fov(self,fov):        
        self._fov =  fov
    
    #Sensor Property    
    @property
    def sensor(self) -> str:
        """The position property"""
        return self._sensor

    @sensor.setter
    def sensor(self,sensor):
        self._sensor =  sensor

        if self.sensor == 'TCS3103-04':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityTCS3103-04.txt")
            print("Responsivity loaded succesfully")                       
        elif self.sensor == 'S10917-35GT':            
            #read text file into NumPy array
            self.responsivity = loadtxt(Constants.SENSOR_PATH+"ResponsivityS10917-35GT.txt")                       
            print("Responsivity loaded succesfully")
        elif self.sensor == '': 
            print("Specify sensor reference")
        else:
            print("Sensor reference not valid")  
    
    def __str__(self) -> str:
        return (
            f'\nList of parameters for photodetector: \n'
            f'Position [x y z]: {self._position} \n'
            f'Normal Vector [x y z]: {self._normal} \n'
            f'Active Area[m2]: {self._area} \n'
            f'FOV: {self._fov} \n'        
            f'Sensor: {self._sensor}'            
        )

    # Plot the spectral responsivity of the photodetector.
    def plot_responsivity(self) -> None:
        plt.plot(self.responsivity[:,0],self.responsivity[:,1],color='r', linestyle='dashed') 
        plt.plot(self.responsivity[:,0],self.responsivity[:,2],color='g', linestyle='dashed') 
        plt.plot(self.responsivity[:,0],self.responsivity[:,3],color='b', linestyle='dashed')
        plt.title("Spectral Responsiity of Photodetector")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Responsivity [A/W]")
        plt.grid()
        plt.show()


#Class for the environment
class Indoorenvironment:        

    # The init method or constructor
    def __init__(self, 
        name: str,
        size: Tuple[float,float,float],
        resolution: float,
        no_reflections: int=3) -> None:
           
        # Instance Variable
        self._name = name    
        self._size = np.array(size)        
        self._resolution = resolution
        self._no_reflections = no_reflections     
        
    
    #Name Property    
    @property
    def name(self):
        """The name property"""        
        return self._name

    @name.setter
    def name(self,value):        
        self._name =  value
    
    #Size Property    
    @property
    def size(self) -> Tuple[float,float,float]:
        """The size property"""
        return self._size

    @size.setter
    def size(self,value):
        print("Set size")
        self._size =  value

    #Number of Reflections Property    
    @property
    def no_reflections(self) -> int:
        """The number of reflections property"""    
        return self._no_reflections

    @no_reflections.setter
    def no_reflections(self,value):        
        self._no_reflections =  value
    
    #Number of Resolution Property    
    @property
    def resolution(self) -> float:
        """The resolution property"""        
        return self._resolution

    @resolution.setter
    def resolution(self,value):        
        self._resolution =  value        
    
    def __str__(self) -> str:
        return(
            f'List of parameters for indoor envirionment:'
            f'Size [x y z] -> [m]: {self._size}'
            f'Order reflection: {self._no_reflections}'            
            f'Resolution points [cm]: {self._resolution}'
        )        
    
    # Set the vector of reflectance at central wavelengths.
    def set_reflectance(self, wall_name, reflectance_wall):
        self._wall_name = wall_name
        self._reflectance_wall = np.array(reflectance_wall)

        if self._wall_name == 'ceiling':
            self._ceiling = self._reflectance_wall
        elif self._wall_name == 'west':
            self._west = self._reflectance_wall
        elif self._wall_name == 'north':
            self._north = self._reflectance_wall
        elif self._wall_name == 'east':
            self._east = self._reflectance_wall
        elif self._wall_name == 'south':
            self._south = self._reflectance_wall
        elif self._wall_name == 'floor':
            self._floor = self._reflectance_wall
        else: 
            print('Invalid wall name.')

    #This function executes the create_grid and computes_parameters methods
    def create_envirorment(self,
        tx_position: Transmitter,
        rx_position: Photodetector,
        tx_normal: Photodetector,
        rx_normal: Photodetector,
        fov: Photodetector) -> None:        

        self.create_grid(tx_position,rx_position,tx_normal,rx_normal)        
        self.compute_parameters(fov)

        return 0

    # Create 3D coordinates of all points in the model
    def create_grid(self,
        tx_position: Tuple[float,float,float],
        rx_position: Tuple[float,float,float],
        tx_normal: Tuple[float,float,float],
        rx_normal: Tuple[float,float,float]) -> None:                

        #Number of ticks in each axis, based on spatial resolution. 
        no_xtick = int(self._size[0]/self._resolution)
        no_ytick = int(self._size[1]/self._resolution)
        no_ztick = int(self._size[2]/self._resolution)

        #print('\nGrid Parameters:')
        #print("Number of ticks [x y z]:",no_xtick,no_ytick,no_ztick)

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
        ceiling_reflectance = np.repeat([self._ceiling],no_xtick*no_ytick,axis=0)
        west_reflectance = np.repeat([self._west],no_ztick*no_xtick,axis=0)
        north_reflectance = np.repeat([self._north],no_ztick*no_ytick,axis=0)
        east_reflectance = np.repeat([self._east],no_ztick*no_xtick,axis=0)
        south_reflectance = np.repeat([self._south],no_ztick*no_ytick,axis=0)
        floor_reflectance = np.repeat([self._floor],no_xtick*no_ytick,axis=0)
               
        #Array with ticks coordinates in every axis
        x_ticks = np.linspace(self._resolution/2,self._size[0]-self._resolution/2,no_xtick)
        y_ticks = np.linspace(self._resolution/2,self._size[1]-self._resolution/2,no_ytick)
        z_ticks = np.linspace(self._resolution/2,self._size[2]-self._resolution/2,no_ztick)

        #Computes the total number of points. If the door is not included, the rx position point is added at end of the array points                
        self.no_points=2*no_xtick*no_ytick + 2*no_ztick*no_xtick + 2*no_ztick*no_ytick   + 2 

        #Generates the x,y,z of grids in each points
        x_ygrid,y_xgrid = np.meshgrid(x_ticks,y_ticks)
        x_zgrid,z_xgrid = np.meshgrid(x_ticks,z_ticks)
        y_zgrid,z_ygrid = np.meshgrid(y_ticks,z_ticks)

        #Save x,y,z coordinates of points in each wall
        ceiling_points[:,0] = floor_points[:,0] = x_ygrid.flatten() 
        ceiling_points[:,1] = floor_points[:,1] = y_xgrid.flatten() 
        ceiling_points[:,2] , floor_points[:,2] = self._size[2] , 0        
        
        west_points[:,0] = east_points[:,0] = x_zgrid.flatten() 
        west_points[:,2] = east_points[:,2] = z_xgrid.flatten() 
        east_points[:,1] , west_points[:,1] = 0 , self._size[1]
        
        north_points[:,1] = south_points[:,1] = y_zgrid.flatten() 
        north_points[:,2] = south_points[:,2] = z_ygrid.flatten() 
        south_points[:,0] , north_points[:,0] = 0 , self._size[0]    
        
        
        #Creates tensors for gridpoints, normal vectors and reflectance vectors.        
        self.gridpoints = torch.from_numpy(np.concatenate((ceiling_points,east_points,south_points,west_points,north_points,floor_points,[tx_position],[rx_position]),axis=0))          
        #self.normal_vectors = torch.from_numpy(np.concatenate((ceiling_normal,east_normal,south_normal,west_normal,north_normal,floor_normal,[Constants.NORMAL_VECTOR_WALL[0]],[Constants.NORMAL_VECTOR_WALL[5]]),axis=0,dtype=np.int8)).reshape(self.no_points,1,3)       
        self.normal_vectors = torch.from_numpy(np.concatenate((ceiling_normal,east_normal,south_normal,west_normal,north_normal,floor_normal,tx_normal,rx_normal),axis=0,dtype=np.float16)).reshape(self.no_points,1,3)       
        self.reflectance_vectors = np.concatenate((ceiling_reflectance,east_reflectance,south_reflectance,west_reflectance,north_reflectance,floor_reflectance,[[0,0,0,0]],[[0,0,0,0]]),axis=0,dtype=np.float16)               
         
        #print("Grid Points->",self.gridpoints)
        

        #Delta area calculation
        self.deltaA = (2*self._size[0]*self._size[1] + 2*self._size[0]*self._size[2] + 2*self._size[1]*self._size[2])/(self.no_points-2)
        
        #print("The total number of points is: ",self.no_points)
        #print("DeltaA: ",self.deltaA)
        print("|>>------------ grid created -------------<<|")


        return 0       

    def compute_parameters(self,fov: float) -> None:
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
        #print("Distance shape->",dist.shape)
        #print("Distance ->",dist)
        
        #Computes the pairwise-difference (vector) using tensor
        diff = -self.gridpoints.unsqueeze(1) + self.gridpoints
        #print("Difference shape->",diff.shape)        
        #print("Difference ->",diff)        
        
        #Computes the unit vector from pairwise-difference usiing tensor
        unit_vector = torch.nan_to_num(torch.div( diff ,dist.reshape(self.no_points,self.no_points,1)),nan=0.0)
        #print("Unitec vector shape ->",unit_vector.shape)

        #Computes the cosine of angle between unit vector and normal vector using tensor.
        cos_phi = torch.sum(unit_vector*self.normal_vectors,dim=2)        
        #print("Cosine shape->",cos_phi.shape)
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

        print("|>>--------- parameters created ----------<<|")
        #np.set_printoptions(threshold=np.inf)        
        #numpy.savetxt("ew_par_dis.csv", ew_par[0,:,:], delimiter=",")  
        #numpy.savetxt("ew_par_cos.csv", ew_par[1,:,:], delimiter=",")  

        return 0

#Class for Recursive Model computations
class Recursivemodel:
    """ This class contains the function to calculates the CIR and DC-gain in the optical channel. """

    # The init method or constructor
    def __init__(self, 
        name: str,
        led: Transmitter, 
        photodetector: Photodetector, 
        room: Indoorenvironment) -> None:
           
        # Instance Variable
        self.name = name
        self.led = led
        self.photodector = photodetector
        self.room = room
    
    # This method simulates the indoor enviornment
    def simulate_channel(self) -> None:
        
        self._compute_cir()
        self._compute_dcgain()
        self._create_spd()
        self._compute_cct_cri()
        self._compute_irradiance()
        self._compute_illuminance()
        self._compute_channelmatrix()

        print("|>>------- Indoor channel simulated ------<<|")       
        
    #Function to compute the CIR
    def _compute_cir(self) -> None:        
        """ Function to compute the channel impulse response for each reflection. 
    
        Parameters:
            led.m: lambertian number to tx emission                         
            led.wall_parameters: 3D array with distance and cosine pairwise-elemets.              
            pd.area: sensitive area in photodetector
            

        Returns: A list with 2d-array [power_ray,time_delay] collection for each 
        refletion [h_0,h_1,...,h_k].
        

        """       
        
        #defing variables and arrays
        tx_index_point = self.room.no_points-2                
        rx_index_point = self.room.no_points-1                
        
        cos_phi = np.zeros((self.room.no_points),dtype=np.float16)
        dis2 = np.zeros((self.room.no_points,self.room.no_points),dtype=np.float16)

        h0_se = np.zeros((self.room.no_points,4),dtype=np.float64)
        h0_er = np.zeros((self.room.no_points,1),dtype=np.float64)                  
                
       
        #Time delay between source and each cells 
        #h0_se[:,1] = room.wall_parameters[0,tx_index_point,:]/SPEED_OF_LIGHT
        #Time delay between receiver and each cells 
        #h0_er[:,1] = room.wall_parameters[0,rx_index_point,:]/SPEED_OF_LIGHT

        #define distance^2 and cos_phi arrays
        dis2 = np.power(self.room.wall_parameters[0,:,:],2)            
        cos_phi = self.room.wall_parameters[1,int(tx_index_point),:]
        #print("COS_PHI for Tx->",cos_phi)
        

        tx_power = (self.led.mlambert+1)/(2*np.pi)*np.multiply(np.divide(1,dis2[tx_index_point,:],out=np.zeros((self.room.no_points)), where=dis2[tx_index_point,:]!=0),np.power(cos_phi,self.led.mlambert))
        rx_wall_factor = pd1.area*self.room.wall_parameters[1,int(rx_index_point),:]

        #Differential power between all grid points without reflectance
        dP_ij = np.zeros((self.room.no_points,self.room.no_points),np.float32)
        dP_ij = np.divide(self.room.deltaA*self.room.wall_parameters[1,:,:]*np.transpose(self.room.wall_parameters[1,:,:]),np.pi*dis2,out=np.zeros_like(dP_ij),where=dis2!=0)         
        #print("Differential Power of Points->",dP_ij)
        
        
        #Array creation for dc_gain and previuos dc_gain
        self.h_k = []
        hlast_er = []
        
        #Array creation for time delay
        self.delay_hk = []
        delay_hlast_er = []

        #Time delay matrix
        tDelay_ij = np.zeros((self.room.no_points,self.room.no_points),dtype=np.float32)
        tDelay_ij = self.room.wall_parameters[0,:,:]/Constants.SPEED_OF_LIGHT
        #print(np.shape(tDelay_ij))


        for i in range(self.room.no_reflections+1):
            
            #Creates the array to save h_k reflections response and last h_er response
            self.h_k.append(np.zeros((self.room.no_points,4),np.float32))
            hlast_er.append(np.zeros((self.room.no_points,4),np.float32)) 

            #Creates the array to save time-delay reflections response and last h_er
            self.delay_hk.append(np.zeros((self.room.no_points,1),np.float32))
            delay_hlast_er.append(np.zeros((self.room.no_points,1),np.float32)) 


            if i == 0:           
                
                #Magnitude of CIR in LoS
                self.h_k[i][0,:] = tx_power[int(rx_index_point)]*rx_wall_factor[int(tx_index_point)]
                
                #Time Delay of CIR in LoS
                self.delay_hk[i][0,0] = tDelay_ij[int(tx_index_point),int(rx_index_point)]
                #print("self.delay_hk->",self.delay_hk[i][0,0])

                print("|>>--------------h{}-computed--------------<<|".format(i))              
                #numpy.savetxt(CIR_PATH+"h0.csv", h_k[i], delimiter=",")
                #print(self.h_k[i])

            elif i==1:

                
                #hlast_er[i] = np.multiply(np.reshape(h0_er[:,0],(-1,1)),self.room.reflectance_vectors)               
                
                #Impulse response between source and each cells without reflectance. The reflectance is added in the h_k computing.
                h0_se = np.multiply(np.reshape(np.multiply(self.room.deltaA*tx_power,self.room.wall_parameters[1,:,int(tx_index_point)]),(-1,1)),self.room.reflectance_vectors)

                #Impulse response between receiver and each cells 
                h0_er[:,0] = np.divide(np.multiply(self.room.wall_parameters[1,:,int(rx_index_point)],rx_wall_factor),np.pi*dis2[rx_index_point,:],out=np.zeros((self.room.no_points)), where=dis2[rx_index_point,:]!=0)
                
                #print("h0_se array->:",h0_se[:,0])
                #print("h0_er array->:",h0_er[:,0])       
                
                #Previous h_er RGBY vectors of magnitude for LoS
                hlast_er[i] = np.repeat(h0_er,repeats=4,axis=1)

                #Current vector for h1 impulse response for RGBY
                #Red-Green-Blue-Yellow                
                self.h_k[i] = np.multiply(h0_se,hlast_er[i])

                #Time-Delay computing
                delay_hlast_er[i] = tDelay_ij[int(rx_index_point),:]
                self.delay_hk[i] = tDelay_ij[int(tx_index_point),:] + delay_hlast_er[i]

                print("|>>--------------h{}-computed--------------<<|".format(i))              
                #np.savetxt(CIR_PATH+"h1.csv", h_k[i], delimiter=","              
                

            elif i>=2:                

                #Time-Delay computing
                delay_hlast_er[i] = np.sum(np.reshape(delay_hlast_er[i-1],(1,-1)) + tDelay_ij,axis=1)/self.room.no_points
                self.delay_hk[i] =  tDelay_ij[int(tx_index_point),:] + delay_hlast_er[i]

                #Computes the last h_er to compute h_k  
                #hlast_er[i] = np.multiply(hlast_er[i-1],np.multiply(self.room.reflectance_vectors,np.reshape(np.sum(dP_ij,axis=0),(-1,1))))
                for color in range(Constants.NO_LEDS):

                    hlast_er[i][:,color] = np.sum(np.multiply(hlast_er[i-1][:,color],np.multiply(self.room.reflectance_vectors[:,color],dP_ij)),axis=1)
                    #print("hlast->",np.shape(hlast_er[i]))

                    self.h_k[i][:,color] = np.multiply(h0_se[:,0],hlast_er[i][:,color])
                    #print("h_k->",np.shape(self.h_k[i]))                 


                


                print("|>>--------------h{}-computed--------------<<|".format(i))              
            

    #This function calculates the total power received from LoS and h_k reflections
    def _compute_dcgain(self) -> None:

        #print("\n Results DC Gain [R G B Y]:")          
        self.h_dcgain = np.zeros((self.room.no_reflections+1,4),np.float32)

        for i in range(0,self.room.no_reflections+1):
            self.h_dcgain[i,:] = np.sum(self.h_k[i][0:-2,0:4], axis = 0) 
            #print(" H"+str(i)+" RGBY DC Gain Power [W]:")          
            #print(self.h_dcgain[i,:])


        self.rgby_dcgain = np.sum(self.h_dcgain, axis = 0)
        #print("DC-gain channel computted")
        #print("Total RGBY DC Gain Power [W]")
        #print(self.rgby_dcgain)

    #Function to create histograms from channel impulse response raw data.
    def _create_histograms(self) -> None:
        """Function to create histograms from channel impulse response raw data. 
        
        The channel impulse response raw data is a list with power and time delay 
        of every ray. Many power histograms are created based on time resolution 
        defined in the TIME_RESOLUTION constant. 

        Parameters:
            h_k: list with channel impulse response [h_0,h_1,...,h_k]. 
            k_reflec: number of reflections
            no_cells: number of points of model

        Returns: A List with the next parameters
            hist_power_time: Power histograms for each reflection
            total_ht: total power CIR histrogram 
            time_scale: 1d-array with time scale

        """

        self.time_resolution = 0.2e-9
        self.bins_hist = 300

        self.total_histogram = np.zeros((self.bins_hist,4))
        self.hist_power_time = []
        delay_aux = np.zeros((room.no_points,1))

        print("//------------- Data report ------------------//")
        print("Time resolution [s]:"+str(self.time_resolution))
        print("Number of Bins:"+str(self.bins_hist))      
        
        delay_los = self.delay_hk[0][0,0]        
        #print(np.shape(delay_los))        
        
        print("Optical power reported in histograms:")

        for k_reflec in range(room.no_reflections+1):            
            
            self.hist_power_time.append(np.zeros((self.bins_hist,4),np.float32))      

            #Delay_aux variable
            delay_aux = np.reshape(self.delay_hk[k_reflec],(-1,1)) - delay_los
            delay_aux = np.floor(delay_aux/self.time_resolution)
            #print(np.shape(delay_aux))

            for j in range(room.no_points):
                #print(int(delay_aux[j]))                
                #print(self.h_k[i][j,:])
                self.hist_power_time[k_reflec][int(delay_aux[j,0]),:] += self.h_k[k_reflec][j,:]

                    
            self.time_scale = linspace(0,self.bins_hist*self.time_resolution,num=self.bins_hist)          
            print("H" + str(k_reflec) + ": " , np.sum(self.hist_power_time[k_reflec],axis=0))       

            self.total_histogram += self.hist_power_time[k_reflec]
        

    #This function plots the channel impulse response for 4 colors
    def _plot_cir(self,channel: str = "") -> None:
        
        self.channel = channel

        if self.channel == 'red':
            color_number = 0
        elif self.channel == 'green':
            color_number = 1
        elif self.channel == 'blue':
            color_number = 2
        elif self.channel == 'yellow':
            color_number = 3
        else:
            print("Invalid color name ('red' or 'green' or 'blue' or 'yellow').")
            color_number = -1        

        if color_number == -1:
            print("Graphs were not generated.")
        else:
            for k_reflec in range(0,room.no_reflections+1):            
                     
                fig, (vax) = plt.subplots(1, 1, figsize=(12, 6))
                
                vax.plot(self.time_scale,self.hist_power_time[k_reflec][:,color_number], 'o',markersize=2)
                vax.vlines(self.time_scale,[0],self.hist_power_time[k_reflec][:,color_number],linewidth=1)

                vax.set_xlabel("time(s) \n Time resolution:",fontsize=15)
                vax.set_ylabel('Power(W)',fontsize=15)
                vax.set_title("Channel "+self.channel+" Impulse Response h"+str(k_reflec)+"(t)",fontsize=20)

                vax.grid(color = 'black', linestyle = '--', linewidth = 0.5)
                
                
                fig.savefig(Constants.REPORT_PATH+"h"+str(k_reflec)+".png")        
                plt.show()


            fig, (vax) = plt.subplots(1, 1, figsize=(12, 6))
            
            vax.plot(self.time_scale,self.total_histogram[:,color_number], 'o',markersize=2)
            vax.vlines(self.time_scale,[0],self.total_histogram[:,color_number],linewidth=1)

            vax.set_xlabel("time(s) \n Time resolution:",fontsize=15)
            vax.set_ylabel('Power(W)',fontsize=15)
            vax.set_title("Channel "+self.channel+" Total Impulse Response h(t)",fontsize=20)

            vax.grid(color = 'black', linestyle = '--', linewidth = 0.5)
            
            
            fig.savefig(Constants.REPORT_PATH+self.channel+"-htotal.png")        
            plt.show()


    #This function creates a SPD of LED from central wavelengths, FWHM and DC gain of channel
    def _create_spd(self) -> None:        
        
        #Array for wavelenght points from 380nm to (782-2)nm with 2nm steps
        self.wavelenght = np.arange(380, 782, 2) 
        
        #Arrays to estimate the RGBY gain spectrum
        self.r_data = self.rgby_dcgain[0]*stats.norm.pdf(self.wavelenght, self.led._wavelengths[0], self.led._fwhm[0]/2)
        self.g_data = self.rgby_dcgain[1]*stats.norm.pdf(self.wavelenght, self.led._wavelengths[1], self.led._fwhm[1]/2)
        self.b_data = self.rgby_dcgain[2]*stats.norm.pdf(self.wavelenght, self.led._wavelengths[2], self.led._fwhm[2]/2)
        self.y_data = self.rgby_dcgain[3]*stats.norm.pdf(self.wavelenght, self.led._wavelengths[3], self.led._fwhm[3]/2)
        self.spd_data = [self.r_data , self.g_data , self.b_data , self.y_data]


    #This function plots the SPD of QLED
    def _plot_spd(self) -> None:

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


    #This function calculates a CCT and CRI of the QLED SPD.
    def _compute_cct_cri(self) -> None:
        
        #Computing the xyz coordinates from SPD-RGBY estimated spectrum
        xyz = lx.spd_to_xyz([self.wavelenght,self.r_data + self.g_data + self.b_data + self.y_data])
        #Computing the CRI coordinates from SPD-RGBY estimated spectrum
        self.cri = lx.cri.spd_to_cri(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]))
        #Computing the CCT coordinates from SPD-RGBY estimated spectrum
        self.cct = lx.xyz_to_cct_ohno2014(xyz)       
        

    #This function calculates the irradiance.
    def _compute_irradiance(self) -> None:                
        self.irradiance = lx.spd_to_power(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]),ptype = 'ru')        


    #This function calculates the illuminance.
    def _compute_illuminance(self) -> None:
        
        self.illuminance = lx.spd_to_power(np.vstack([self.wavelenght,(self.r_data + self.g_data + self.b_data + self.y_data)/pd1.area]),ptype = 'pu')         


    #This function computes channel matrix
    def _compute_channelmatrix(self) -> None:

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
                

if __name__ == "__main__":

    starttime = timeit.default_timer()

    #code to simulate a VLC channel

    led1 = Transmitter("Led1",position=[2.5,2.5,3],normal=[0,0,-1],mlambert=1,power=1,wavelengths=[650,530,430,580],fwhm=[20,12,20,20])    
    led1.led_pattern()

    pd1 =  Photodetector("PD1",position=[0.5,1.0,0],normal=[0,0,1],area=1e-4,fov=85,sensor='S10917-35GT')    
    pd1.plot_responsivity()
    

    room = Indoorenvironment("Room",size=[5,5,3],no_reflections=3,resolution=1/4)    
    room.set_reflectance('ceiling',[0.8,0.8,0.8,0.8])
    room.set_reflectance('west',[0.8,0.8,0.8,0.8])
    room.set_reflectance('north',[0.8,0.8,0.8,0.8])
    room.set_reflectance('east',[0.8,0.8,0.8,0.8])
    room.set_reflectance('south',[0.8,0.8,0.8,0.8])
    room.set_reflectance('floor',[0.3,0.3,0.3,0.3])    
    room.create_envirorment(led1._position,pd1._position,led1._normal,pd1._normal,pd1._fov)
    

    channel_model = Recursivemodel("ChannelModelA",led1,pd1,room)
    channel_model.simulate_channel()
    

    #ending code


    print("\nThe execution time is :", timeit.default_timer() - starttime)
    print("Simulation finished.")