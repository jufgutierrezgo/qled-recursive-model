# RecursiveModel-VLC

The **qled-recursive-model * is a script to compute the DC-gain of optical link in the visible range using a recursive model presented in [1]. 

## Installation

Make sure have installed all libraries used in the python script. 

## Usage

To run this recursive model you only need run the qled_csk_recursive.py. Once the script execution has finished, the DC-gain for four color LEDs is computed. 

This model uses a set of parameters to define features of source, receiver and environment, as follows:

```python
#defining a new object type Transmitter named Led1
led1 = Transmitter("Led1")
#Set the position of the LED
led1.set_position([2.5,2.5,3])
#Set the normal vector of LED
led1.set_normal([0,0,-1]) 
#Set the Lambert number of LED
led1.set_mlambert(1)
#Set the power of LED in watts
led1.set_power(1)
#Set the central wavelength of four color LEDs
led1.set_wavelengths([650,530,430,580])
#Set the FWHM of four color LEDs
led1.set_fwhm([20,12,20,20])
#Set print paremeters
led1.get_parameters()
#Set the spatial power distribution of LED
led1.led_pattern()

#defining a new object type Photodetector named Led1
pd1 =  Photodetector("PD1")
#set the position of the photodetector
pd1.set_position([0.5,1.0,0])
#Set the normal vector of the detector
pd1.set_normal([0,0,1])
#Set the active area of detector
pd1.set_area(1e-4)
#Set the FOV 
pd1.set_fov(85)
#Set spectral responsivity using a predifened sensors
pd1.set_responsivity('S10917-35GT')
#Plot responsivity 
pd1.plot_responsivity()
#Print parameters
pd1.get_parameters()

#defining a new object type Indoorenviorment named Room
room = Indoorenvironment("Room")
#Set size of the indoor space
room.set_size([5,5,3])
#Set the numer of refelection to compute the DC-gain
room.set_noreflections(3)
#Set the point resolution of the camputation
room.set_pointresolution(1/8)
#Set the reflectance of each wall at four central wavelengths
room.set_reflectance('ceiling',[0.8,0.8,0.8,0.8])
room.set_reflectance('west',[0.8,0.8,0.8,0.8])
room.set_reflectance('north',[0.8,0.8,0.8,0.8])
room.set_reflectance('east',[0.8,0.8,0.8,0.8])
room.set_reflectance('south',[0.8,0.8,0.8,0.8])
room.set_reflectance('floor',[0.3,0.3,0.3,0.3])    
#Create grid of the model
room.create_grid(led1.position,pd1.position)
#Create parameters of model (pairwise cosine and distance)
room.create_parameters(pd1.fov)

```

The previous values for source, receiver and environment are the default values. Using this set of parameters, the script compute the DC-gain of the channel, the lighting performance index and the interchannel matrix using the follow functions:

```python
#Defining a new object type REcursivemodel named ChanelModelA
channel_model = Recursivemodel("ChannelModelA",led1,pd1,room)
#Compute the channel impulse response
channel_model.compute_cir()
#Compute the DC channel gain
channel_model.compute_dcgain()
#Compute spectral power distribution 
channel_model.create_spd()
#Plot spectral power distribution 
channel_model.plot_spd()
#Compute Correlated Color Temperature and Color Rendering index 
channel_model.compute_cct_cri()
#Compute irradiance [W/m2]
channel_model.compute_irradiance()
#Compute irradiance [Lux]
channel_model.compute_illuminance()
#Compute inter channel interference matrix 
channel_model.compute_channelmatrix()
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Referencies
[1] Barry, J. R., Kahn, J. M., Krause, W. J., Lee, E. A., & Messerschmitt, D. G. (1993). Simulation of multipath impulse response for indoor wireless optical channels. IEEE journal on selected areas in communications, 11(3), 367-379.
