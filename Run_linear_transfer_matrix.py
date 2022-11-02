import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt


####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCpu(omp_num_threads=5)
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

buf = context.new_buffer()


#num_turns = int(1e2)


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

#%% 
##################
# RF CAVITY #
##################

#fname_sequence = '/home/pkruyt/anaconda3/lib/python3.9/site-packages/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'

fname_sequence ='/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'



with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xt.Line.from_dict(input_data['line'])

#%% 
##################
# Laser Cooler #
##################

#sigma_dp = 2e-4 # relative ion momentum spread

#bunch_intensity = 1e11
sigma_z = 22.5e-2
nemitt_x = 2e-6
nemitt_y = 2.5e-6

#sigma_dp = sigma_z / beta
sigma_dp = 2e-4 # relative ion momentum spread

#laser-ion beam collision angle
theta_l = 2.6*np.pi/180 # rad
theta_l = 0
nx = 0; ny = -np.sin(theta_l); nz = -np.cos(theta_l)

# Ion excitation energy:
hw0 = 230.823 # eV
hc = 0.19732697e-6 # eV*m (ħc)
lambda_0 = 2*np.pi*hc/hw0 # m -- ion excitation wavelength

lambda_l = lambda_0*gamma*(1 + beta*np.cos(theta_l)) # m -- laser wavelength

# Shift laser wavelength for fast longitudinal cooling:
lambda_l = lambda_l*(1+sigma_dp) # m

laser_frequency = c/lambda_l # Hz
sigma_w = 2*np.pi*laser_frequency*sigma_dp
#sigma_w = 2*np.pi*laser_frequency*sigma_dp/2 # for fast longitudinal cooling

sigma_t = 1/sigma_w # sec -- Fourier-limited laser pulse
print('Laser pulse duration sigma_t = %.2f ps' % (sigma_t/1e-12))

print('Laser wavelength = %.2f nm' % (lambda_l/1e-9))

laser_waist_radius = 1.3e-3

GF_IP = xt.IonLaserIP(_buffer=buf,
                      laser_direction_nx = 0,
                      laser_direction_ny = ny,
                      laser_direction_nz = nz,
                      laser_energy         = 5e-3, # J
                      laser_duration_sigma = sigma_t, # sec
                      laser_wavelength = lambda_l, # m
                      laser_waist_radius = 1.3e-3, # m
                      ion_excitation_energy = hw0, # eV
                      ion_excited_lifetime  = 76.6e-12, # sec
   
                        
   )

#%%




#%%
##################
#  Import Twiss  #
##################

with open('SPS_lin.json', 'r') as fid:
    loaded_dct = json.load(fid)
SPS_lin = xt.Line.from_dict(loaded_dct)







# Load particles from json file to selected context
with open('particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)
    
import pickle

a_file = open("twiss.pkl", "rb")

twiss = pickle.load(a_file)    

    
#%%
##################
#     Tracking   #
##################



disp_x_0=5


arc=xt.LinearTransferMatrix(Q_x=twiss['qx'], Q_y=twiss['qy'],
beta_x_0=twiss['betx'][0], beta_x_1=twiss['betx'][-1], beta_y_0=twiss['bety'][0], beta_y_1=twiss['bety'][-1],
alpha_x_0=twiss['alfx'][0], alpha_x_1=twiss['alfx'][-1], alpha_y_0=twiss['alfy'][0], alpha_y_1=twiss['alfy'][-1],
disp_x_0=disp_x_0, disp_x_1=disp_x_0, disp_y_0=0, disp_y_1=0,
Q_s=twiss['qs'], beta_s=twiss['betz0'],
chroma_x=twiss['dqx'], chroma_y=twiss['dqy'])


SPS_lin = xt.Line()

SPS_lin.append_element(arc,'SPS_LinearTransferMatrix')
for i in range(1):
        SPS_lin.append_element(GF_IP, f'GammaFactory_IP{i}')



num_turns=int(1e4)

lin_tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)


import datetime
first_time = datetime.datetime.now()

lin_tracker.track(particles0, num_turns=num_turns, turn_by_turn_monitor=True)




excited = (particles0.state == 2)
true=any(excited)

print('true',true)

aa=SPS_lin.particle_ref


#%%
##################
#    Emmitance   #
##################

x = lin_tracker.record_last_track.x
px = lin_tracker.record_last_track.px

y = lin_tracker.record_last_track.y
py = lin_tracker.record_last_track.py

zeta = lin_tracker.record_last_track.zeta
delta = lin_tracker.record_last_track.delta


#%%
################
# Phase Space #
################
from acc_lib import plot_tools
import matplotlib.pyplot as plt




fig1=plt.figure()


plot_tools.plot_phase_space_ellipse(fig1, lin_tracker, axis='horizontal')

fig1.suptitle(f'Phase Space disp_x={disp_x_0}')

x=lin_tracker.record_last_track.x

xturn=x[:,1]

#%%

def emittance_2d(tracker):
    
    
    
    x=tracker.record_last_track.x
    px=tracker.record_last_track.px
    
    delta=tracker.record_last_track.delta
    
    x=x-disp_x_0*delta
    
    cov_list=[]
    for i in range(num_turns):
        x0=x[:,i]
        px0=px[:,i]
    
    
        cov00=np.cov(x0,px0)
    
        det00 = np.sqrt((np.linalg.det(cov00)))*beta*gamma
        cov_list.append(det00)
        
   
    
    plt.figure()
    plt.plot(cov_list)
    
    plt.title('Horizontal emmitance vs turns (2d)')
    plt.ylabel('emittance (m)')
    plt.xlabel('number of turns')    


def emittance_6d(tracker):
    
    x=tracker.record_last_track.x
    px=tracker.record_last_track.px
    y=tracker.record_last_track.y
    py=tracker.record_last_track.py
    zeta=tracker.record_last_track.zeta
    delta=tracker.record_last_track.delta
    
    
    cov_list=[]
    for i in range(num_turns):
        x0=x[:,i]
        px0=px[:,i]
                      
        y0=y[:,i]
        py0=py[:,i]
               
        zeta0=zeta[:,i]
        pdelta0=delta[:,i]
    
        
        
        data = np.array([x0,px0,
                         y0,py0,
                         zeta0,pdelta0])
        
    
        cov00=np.cov(data)
    
        det00 = np.sqrt((np.linalg.det(cov00)))*beta*gamma
        cov_list.append(det00)
        
   
    
    plt.figure()
    plt.plot(cov_list)
    
    plt.title('Total emmitance vs turns (6d)')
    plt.ylabel('emittance $(m^3)$')
    plt.xlabel('number of turns')    




emittance_2d(lin_tracker)
emittance_6d(lin_tracker)




#%%



