import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp

import matplotlib.pyplot as plt
from tqdm import tqdm
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



SPS_nonlin_tracker = xt.Tracker(_context=context, _buffer=buf, line=sequence)

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
##################
#  Import Twiss  #
##################

# Load particles from json file to selected context
with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

particles0.delta=0.0000001



particles0.zeta=0.0001
# particles0.x=0
# particles0.px=0
# particles0.y=0
# particles0.py=0

particles_old=particles0.copy()
particles00=particles0.copy()
    
import pickle

a_file = open("cache/twiss.pkl", "rb")

twiss = pickle.load(a_file)    

#from sps line
frequency=200266000.0
lag=180
voltage=3000000.0

cavity=xt.Cavity(voltage=frequency,frequency=voltage,lag=lag)
   
cavity_index=5408
    
#%%



arc=xt.LinearTransferMatrix(Q_x=twiss['qx'], Q_y=twiss['qy'],
beta_x_0=twiss['betx'][0], beta_x_1=twiss['betx'][-1],
beta_y_0=twiss['bety'][0], beta_y_1=twiss['bety'][-1],
alpha_x_0=twiss['alfx'][0], alpha_x_1=twiss['alfx'][-1],
alpha_y_0=twiss['alfy'][0], alpha_y_1=twiss['alfy'][-1],
disp_x_0=twiss['dx'][0], disp_x_1=twiss['dx'][-1],
disp_y_0=twiss['dy'][0], disp_y_1=twiss['dy'][-1],
beta_s=twiss['betz0'],
Q_s=-twiss['qs'],
#lag=0*np.pi,
chroma_x=twiss['dqx'], chroma_y=twiss['dqy'])


SPS_lin = xt.Line()

SPS_lin.append_element(arc,'arc')


#%%

num_turns=int(1e3)



SPS_lin_tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)


SPS_lin_tracker.track(particles0, num_turns=num_turns, turn_by_turn_monitor=True)
SPS_nonlin_tracker.track(particles00, num_turns=num_turns, turn_by_turn_monitor=True)


#%%
#X

x_lin = SPS_lin_tracker.record_last_track.x
px_lin = SPS_lin_tracker.record_last_track.px

y_lin = SPS_lin_tracker.record_last_track.y
py_lin = SPS_lin_tracker.record_last_track.py

zeta_lin = SPS_lin_tracker.record_last_track.zeta
delta_lin = SPS_lin_tracker.record_last_track.delta


#################################################################################


x_nonlin = SPS_nonlin_tracker.record_last_track.x
px_nonlin = SPS_nonlin_tracker.record_last_track.px

y_nonlin = SPS_nonlin_tracker.record_last_track.y
py_nonlin = SPS_nonlin_tracker.record_last_track.py

zeta_nonlin = SPS_nonlin_tracker.record_last_track.zeta
delta_nonlin = SPS_nonlin_tracker.record_last_track.delta


#%%

# np.save('cache/comparison/x_lin.npy', x_lin)
# np.save('cache/comparison/px_lin.npy', px_lin)
# np.save('cache/comparison/y_lin.npy', y_lin)
# np.save('cache/comparison/py_lin.npy', py_lin)
# np.save('cache/comparison/zeta_lin.npy', zeta_lin)
# np.save('cache/comparison/delta_lin.npy', delta_lin)

# np.save('cache/comparison/x_nonlin.npy', x_nonlin)
# np.save('cache/comparison/px_nonlin.npy', px_nonlin)
# np.save('cache/comparison/y_nonlin.npy', y_nonlin)
# np.save('cache/comparison/py_nonlin.npy', py_nonlin)
# np.save('cache/comparison/zeta_nonlin.npy', zeta_nonlin)
# np.save('cache/comparison/delta_nonlin.npy', delta_nonlin)


#%%

# diff_x=np.subtract(x_lin.tolist()[0],x_nonlin.tolist()[0])

# plt.figure()
# plt.title('linear x vs nonlinear x')
# plt.plot(x_lin.tolist()[0],label='linear')
# plt.plot(x_nonlin.tolist()[0],label='nonlinear')
# plt.legend(loc='best')
# plt.ylabel('x(m)')
# plt.xlabel('number of turns')
# plt.show()

# plt.figure()
# plt.title('difference between linear and nonlinear x')

# plt.plot(diff_x/np.max(x_nonlin)*100)
# plt.ylabel('%$ \Delta x$')
# plt.xlabel('number of turns')
# plt.show()


# #%%

# diff_y=np.subtract(y_lin.tolist()[0],y_nonlin.tolist()[0])

# plt.figure()
# plt.title('linear y vs nonlinear y')
# plt.plot(y_lin.tolist()[0],label='linear')
# plt.plot(y_nonlin.tolist()[0],label='nonlinear')
# plt.legend(loc='best')
# plt.ylabel('y(m)')
# plt.xlabel('number of turns')
# plt.show()

# plt.figure()
# plt.title('difference between linear and nonlinear y')

# plt.plot(diff_y/np.max(y_nonlin)*100)
# plt.ylabel('%$ \Delta y$')
# plt.xlabel('number of turns')
# plt.show()


# #%%

# diff_px=np.subtract(px_lin.tolist()[0],px_nonlin.tolist()[0])

# plt.figure()
# plt.title('linear px vs nonlinear px')
# plt.plot(px_lin.tolist()[0],label='linear')
# plt.plot(px_nonlin.tolist()[0],label='nonlinear')
# plt.legend(loc='best')
# plt.ylabel('px(m)')
# plt.xlabel('number of turns')
# plt.show()

# plt.figure()
# plt.title('difference between linear and nonlinear px')

# plt.plot(diff_px/np.max(px_nonlin)*100)
# plt.ylabel('%$ \Delta px$')
# plt.xlabel('number of turns')
# plt.show()

# #%%

# diff_py=np.subtract(py_lin.tolist()[0],py_nonlin.tolist()[0])

# plt.figure()
# plt.title('linear py vs nonlinear py')
# plt.plot(py_lin.tolist()[0],label='linear')
# plt.plot(py_nonlin.tolist()[0],label='nonlinear')
# plt.legend(loc='best')
# plt.ylabel('py(m)')
# plt.xlabel('number of turns')
# plt.show()

# plt.figure()
# plt.title('difference between linear and nonlinear py')

# plt.plot(diff_py/np.max(py_nonlin)*100)
# plt.ylabel('%$ \Delta py$')
# plt.xlabel('number of turns')
# plt.show()

#%%

diff_zeta=np.subtract(zeta_lin.tolist()[0],zeta_nonlin.tolist()[0])

plt.figure()
plt.title('linear zeta vs nonlinear zeta')

plt.plot(zeta_nonlin.tolist()[0],label='nonlinear')
plt.plot(zeta_lin.tolist()[0],label='linear')
plt.legend(loc='best')
plt.ylabel('zeta(m)')
plt.xlabel('number of turns')
plt.show()

plt.figure()
plt.title('difference between linear and nonlinear zeta')

plt.plot(diff_zeta/np.max(zeta_nonlin)*100)
plt.ylabel('%$ \Delta zeta$')
plt.xlabel('number of turns')
plt.show()
