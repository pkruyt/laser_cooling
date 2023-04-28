import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp

from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy import constants 
####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCpu(omp_num_threads=5)
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

buf = context.new_buffer()


# Ion properties:
m_u = 931.49410242e6 # eV/c^2 -- atomic mass unit
A = 207.98 # Lead-208
Z = 82  # Number of protons in the ion (Lead)
Ne = 3 # Number of remaining electrons (Lithium-like)
m_e = 0.511e6 # eV/c^2 -- electron mass
m_p = 938.272088e6 # eV/c^2 -- proton mass
c = 299792458.0 # m/s

m_ion = A*m_u + Ne*m_e # eV/c^2

equiv_proton_momentum = 236e9 # eV/c = gamma_p*m_p*v

gamma_p = np.sqrt( 1 + (equiv_proton_momentum/m_p)**2 ) # equvalent gamma for protons in the ring


p0c = equiv_proton_momentum*(Z-Ne) # eV/c
gamma = np.sqrt( 1 + (p0c/m_ion)**2 ) # ion relativistic factor
beta = np.sqrt(1-1/(gamma*gamma)) # ion beta



# Load particles from json file to selected context
with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

particles_old=particles0.copy()


max_list=[]


laser_wavelength =     1.0399999989774836e-06
max_laser_wavelength = 1.0340260500668464e-06

laser_range=np.arange(1.03*1e-6,1.04*1e-6,1e-15)
laser_range = np.arange(max_laser_wavelength - 0.0051e-6, max_laser_wavelength + 0.0051e-6, 1e-16)


for laser_wavelength in tqdm(laser_range):
    gamma_emit=1/76.6e-12
    import math
    #laser_omega_ion_frame = 
    hw0 = 230.823 # eV
    eV = 1.602176634e-19#; // J
    hbar = 1.054571817e-34#; // J*sec
    ion_excitation_energy=hw0
    OmegaTransition = ion_excitation_energy*eV/hbar# // rad/sec #OmegaTransition= 350681870336557632.000000
    
    cos_theta = 1
    laser_wavelength = laser_wavelength
    
    
    p0c=18644000000000.0
    m0=193733676421.31158
    pc = p0c*(1.0+particles_old.delta)#; // eV
    gamma = np.sqrt(1.0 + pc*pc/(m0*m0));
    
    laser_omega_ion_frame = (2.0*math.pi*c/laser_wavelength)*(1.0+beta*cos_theta)*gamma;
    
    
    excitation_probability = gamma_emit/(2*math.pi)/((laser_omega_ion_frame-OmegaTransition)**2- (0.5*gamma_emit)**2)
    
    max_prob = max(excitation_probability)
    max_list.append(max_prob)

max_index = np.argmax(max_list)
max_wavelength = laser_range[max_index] #1.03393088959806e-06
    
plt.plot(laser_range,max_list)    
plt.title('Maximum excitation probability vs laser wavelength')
plt.xlabel('laser wavelength [m]')
plt.ylabel('maximum excitation probability ')