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
import os
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


a_file = open("../cache/twiss.pkl", "rb")

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
with open('../cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

particles_old=particles0.copy()

x_old=particles_old.x


# with open('cache/particles_new.json', 'r') as fid:
#     particles0= xp.Particles.from_dict(json.load(fid), _context=context)

std_delta = particles_old.delta.std()

num_particles=len(particles0.x)    

#%% 
##################
# Laser Cooler #
##################

sigma_dp = 2e-4 # relative ion momentum spread
#sigma_dp = std_delta

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
#laser_x=0.0010000
laser_energy=5e-3, # J



# variable_name='laser_x'
# variable_name='laser_wavelength'
# variable_name='sigma_t'
# variable_name='laser_waist_radius'
#variable_name='laser_energy'
variable_name='theta'

for i in tqdm(np.arange(start=0,stop=4,step=1)):  
#for i in tqdm(np.arange(start=0.8,stop=1.2,step=0.1)): 
    lambda_l_sweep=[lambda_l*(1+i*sigma_dp)]
    laser_x_sweep=[laser_x*(i)]
    sigma_t_sweep=[sigma_t*(i)]
    laser_waist_radius_sweep=[laser_waist_radius*(i)]
    laser_energy_sweep=[laser_energy*(i)]
        
    theta_l = 10
    theta_sweep=[theta_l*(i)]
    nx = 0; ny = -np.sin(theta_sweep); nz = -np.cos(theta_sweep)
    
    
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',i)
    
    
    SPS_lin = xt.Line()
    SPS_lin.append_element(arc,'SPS_LinearTransferMatrix')
    
    GF_IP = xt.IonLaserIP(_buffer=buf,
                          laser_x=laser_x,
                          
                          laser_direction_nx = nx,
                          laser_direction_ny = ny,
                          laser_direction_nz = nz,
                          laser_energy         = laser_energy, # J
                          laser_duration_sigma = sigma_t, # sec
                          laser_wavelength = lambda_l, # m
                          laser_waist_radius = laser_waist_radius, # m
                          ion_excitation_energy = hw0, # eV
                          ion_excited_lifetime  = 76.6e-12, # sec
                              
       )
    
    
    SPS_lin.append_element(GF_IP, 'GammaFactory_IP')
    
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
    
    
    #SPS_lin.append_element(skew_quad, 'skew_quad')
          
    #%%
        
    num_turns=int(1e4) #4e4 is maximum
    
    tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)
    
    
    
    monitor = xt.ParticlesMonitor(_context=context,
                                  start_at_turn=0, stop_at_turn=num_turns,
                                  #n_repetitions=3,      # <--
                                  #repetition_period=20, # <--
                                  num_particles=num_particles)
   
        
    # Set the path for the directory that will be created
    home = '/home/pkruyt/Documents/'
    path = os.path.join(home, f'sweep_{variable_name}')
    
    # Create the directory if it does not already exist
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    
    # Define a list of variables that will be memory-mapped
    vars_to_memmap = ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state']
    
    # Initialize an empty dictionary to hold the memory-mapped arrays
    fp_vars = {}
    
    # Create memory-mapped arrays for each variable in vars_to_memmap
    for var in vars_to_memmap:
        fp_var = np.memmap(os.path.join(path, f'{var}.npy'), dtype=np.float64, mode='w+', shape=(num_particles, num_turns))
        fp_vars[var] = fp_var
    
    # Create a copy of the particles_old array before starting the tracking
    particles0 = particles_old.copy()
    
    # Loop over the number of turns
    for iturn in range(num_turns):
        # Run the tracker on the particles0 array
        tracker.track(particles0)
        
        # Loop over the variables in vars_to_memmap and populate the corresponding
        # memory-mapped arrays with the attribute data from the particles0 array
        for var in vars_to_memmap:
            fp_vars[var][:, iturn] = getattr(particles0, var)
            
    
    
    particles_old_vars = {}  # Dictionary to store the values of variables in the particles_old object

    for var in vars_to_memmap:      
        # Retrieve the value of the current variable from the particles_old object
        particles_old_var = getattr(particles_old, var)
        # Expand the dimensions of the variable value to include an additional dimension along the axis=1 axis
        particles_old_var_expanded = np.expand_dims(particles_old_var, axis=1)
        # Add the expanded variable value to the particles_old_vars dictionary using the variable name as the key
        particles_old_vars[var] = particles_old_var_expanded
    
        
    # Create the directory if it does not already exist 
    path=home+f'sweep_{variable_name}/{variable_name}:{i}'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
        
        
    variables={} # Dictionary to store the variables x,px,y,py,zeta,delta,and state
    # Concatenate data from particles_old and fp_vars
    for var in vars_to_memmap:
        var_old = particles_old_vars[var]
        var_new = fp_vars[var]
        var_concat = np.append(var_old, var_new, axis=1)
        np.save(home+f'sweep_{variable_name}/{variable_name}:{i}/{var}:{i}.npy', var_concat)
        variables[var]=var_concat
        # Assign the concatenated array to the appropriate variable
        exec(f'{var} = var_concat')
        
        
    import emittance    
        
    x=variables['x']
    px=variables['px']
    y=variables['y']
    py=variables['py']
    zeta=variables['zeta']
    delta=variables['delta']
    state=variables['state']
           
    
    emitt_z = emittance.emittance_2d(zeta, delta)
    emitt_x = emittance.emittance_2d(x, px)
    emitt_y = emittance.emittance_2d(y, py)
    
    path=home+f'sweep_{variable_name}/emittance_results/'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

    path=home+f'sweep_{variable_name}/emittance_results/emitt_z/'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    path=home+f'sweep_{variable_name}/emittance_results/emitt_x/'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)    
        
        
        
    np.save(home+f'sweep_{variable_name}/emittance_results/emitt_z/emitt_z:{i}.npy', emitt_z)   
    np.save(home+f'sweep_{variable_name}/emittance_results/emitt_x/emitt_x:{i}.npy', emitt_x)   
    
    
    for var in vars_to_memmap:
        del fp_vars[var] 

