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

#%% 
##################
# RF CAVITY #
##################

#fname_sequence = '/home/pkruyt/anaconda3/lib/python3.9/site-packages/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'

fname_sequence ='/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'



with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xt.Line.from_dict(input_data['line'])


a_file = open("cache/twiss.pkl", "rb")

twiss = pickle.load(a_file)    

arc=xt.LinearTransferMatrix(Q_x=twiss['qx'], Q_y=twiss['qy'],
beta_x_0=twiss['betx'][0], beta_x_1=twiss['betx'][-1],
beta_y_0=twiss['bety'][0], beta_y_1=twiss['bety'][-1],
alpha_x_0=twiss['alfx'][0], alpha_x_1=twiss['alfx'][-1],
alpha_y_0=twiss['alfy'][0], alpha_y_1=twiss['alfy'][-1],
disp_x_0=twiss['dx'][0], disp_x_1=twiss['dx'][-1],
disp_y_0=twiss['dy'][0], disp_y_1=twiss['dy'][-1],
beta_s=twiss['betz0'],
Q_s=-twiss['qs'],
chroma_x=twiss['dqx'], chroma_y=twiss['dqy'])

SPS_lin = xt.Line()


SPS_lin.append_element(arc,'SPS_LinearTransferMatrix')

# Load particles from json file to selected context
with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

particles_old=particles0.copy()



# with open('cache/particles_new.json', 'r') as fid:
#     particles0= xp.Particles.from_dict(json.load(fid), _context=context)

std_delta = particles_old.delta.std()

num_particles=len(particles0.x)    

#%% 
##################
# Laser Cooler #
##################

sigma_dp = 2e-4 # relative ion momentum spread
sigma_dp = 2e-4 

#laser-ion beam collision angle
theta_l = 2.6*np.pi/180 # rad
theta_l = 0
nx = 0; ny = -np.sin(theta_l); nz = -np.cos(theta_l)

# Ion excitation energy:
hw0 = 230.823 # eV
hc = 0.19732697e-6 # eV*m (??c)
lambda_0 = 2*np.pi*hc/hw0 # m -- ion excitation wavelength

lambda_l = lambda_0*gamma*(1 + beta*np.cos(theta_l)) # m -- laser wavelength

# Shift laser wavelength for fast longitudinal cooling:
lambda_l = lambda_l*(1+1*sigma_dp) # m

laser_frequency = c/lambda_l # Hz
sigma_w = 2*np.pi*laser_frequency*sigma_dp
#sigma_w = 2*np.pi*laser_frequency*sigma_dp/2 # for fast longitudinal cooling

sigma_t = 1/sigma_w # sec -- Fourier-limited laser pulse
print('Laser pulse duration sigma_t = %.2f ps' % (sigma_t/1e-12))

print('Laser wavelength = %.2f nm' % (lambda_l/1e-9))

laser_waist_radius = 1.3e-3
#laser_waist_radius = 1.3e-7

laser_x=0.0020000

GF_IP = xt.IonLaserIP(_buffer=buf,
                      laser_x=laser_x,
                      
                      laser_direction_nx = 0,
                      laser_direction_ny = 0,
                      laser_direction_nz = -1,
                      laser_energy         = 5e-3, # J
                      laser_duration_sigma = sigma_t, # sec
                      laser_wavelength = lambda_l, # m
                      laser_waist_radius = laser_waist_radius, # m
                      ion_excitation_energy = hw0, # eV
                      ion_excited_lifetime  = 76.6e-12, # sec
                          
   )



                    
 


for i in range(1):
        SPS_lin.append_element(GF_IP, f'GammaFactory_IP{i}')

#%%

########################
#    Skew-quadrupole   #
########################

Lsq = 1.0 # m
Ksq = 0.001 # 1/m^2
Ksq = 0.01 # 1/m^2

#Ksq = 0.00001 # 1/m^2

k  = np.sqrt(Ksq) # 1/m
kl = k*Lsq




skew_quad=xt.Multipole(order=0, 
                       #knl=None,
                       ksl=[0,Ksq],
                       length=Lsq)



for i in range(1):
        SPS_lin.append_element(skew_quad, f'skew_quad{i}')




#%%


num_turns=int(1e2) #4e4 is maximum

tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)



monitor = xt.ParticlesMonitor(_context=context,
                              start_at_turn=0, stop_at_turn=num_turns,
                              #n_repetitions=3,      # <--
                              #repetition_period=20, # <--
                              num_particles=num_particles)

num_cycles=int(1e4)


for i in tqdm(range(num_cycles)):

    fp_x = np.memmap('/home/pkruyt/Documents/cache_memory/x.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    fp_px = np.memmap('/home/pkruyt/Documents/cache_memory/px.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    
    fp_y = np.memmap('/home/pkruyt/Documents/cache_memory/y.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    fp_py = np.memmap('/home/pkruyt/Documents/cache_memory/py.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    
    fp_zeta = np.memmap('/home/pkruyt/Documents/cache_memory/zeta.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    fp_delta = np.memmap('/home/pkruyt/Documents/cache_memory/delta.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    
    fp_state = np.memmap('/home/pkruyt/Documents/cache_memory/state.npy', dtype=np.float64, mode='w+', shape=(num_particles,num_turns))
    
    for iturn in (range(num_turns)):
        #monitor.track(particles0)
        tracker.track(particles0)
        
        fp_x[:, iturn] = particles0.x
        fp_px[:, iturn] = particles0.px
    
        
        fp_y[:, iturn] = particles0.y
        fp_py[:, iturn] = particles0.py
        
        fp_zeta[:, iturn] = particles0.zeta
        fp_delta[:, iturn] = particles0.delta
        fp_state[:, iturn] = particles0.state
    
    x = fp_x[:, -1]    
    px = fp_px[:, -1]    
    
    y = fp_y[:, -1]    
    py = fp_py[:, -1] 
    
    zeta = fp_zeta[:, -1]    
    delta = fp_delta[:, -1] 
    
    state = fp_state[:, -1] 
    
    np.save(f'/home/pkruyt/Documents/cache_memory/x{i}.npy', x)
    np.save(f'/home/pkruyt/Documents/cache_memory/px{i}.npy', px)
    
    np.save(f'/home/pkruyt/Documents/cache_memory/y{i}.npy', y)
    np.save(f'/home/pkruyt/Documents/cache_memory/py{i}.npy', py)
    
    np.save(f'/home/pkruyt/Documents/cache_memory/zeta{i}.npy', zeta)
    np.save(f'/home/pkruyt/Documents/cache_memory/delta{i}.npy', delta)
    
    np.save(f'/home/pkruyt/Documents/cache_memory/state{i}.npy', state)
        
    del fp_x
    del fp_px
    
    del fp_y
    del fp_py
    
    del fp_zeta
    del fp_delta
    
    del fp_state



particles_new = particles0.copy()

# with open('cache/particles_new.json', 'w') as fid:
#     json.dump(particles_new.to_dict(), fid, cls=xo.JEncoder)

