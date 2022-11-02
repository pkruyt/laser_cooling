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




#%%
##################
#  Import Twiss  #
##################

with open('cache/SPS_lin.json', 'r') as fid:
    loaded_dct = json.load(fid)
SPS_lin = xt.Line.from_dict(loaded_dct)



# Load particles from json file to selected context
with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)
    
import pickle

a_file = open("cache/twiss.pkl", "rb")

twiss = pickle.load(a_file)    

    
#%%
##################
#     Tracking   #
##################

particles_old=particles0.copy()
particles0.delta = abs(particles_old.delta)


disp_x_0=0.5


arc=xt.LinearTransferMatrix(Q_x=twiss['qx'], Q_y=twiss['qy'],
beta_x_0=twiss['betx'][0], beta_x_1=twiss['betx'][-1], beta_y_0=twiss['bety'][0], beta_y_1=twiss['bety'][-1],
alpha_x_0=twiss['alfx'][0], alpha_x_1=twiss['alfx'][-1], alpha_y_0=twiss['alfy'][0], alpha_y_1=twiss['alfy'][-1],
disp_x_0=disp_x_0, disp_x_1=disp_x_0, disp_y_0=0, disp_y_1=0,
Q_s=0, beta_s=twiss['betz0'],
chroma_x=twiss['dqx'], chroma_y=twiss['dqy'])


SPS_lin = xt.Line()

SPS_lin.append_element(arc,'SPS_LinearTransferMatrix')
# for i in range(1):
#         SPS_lin.append_element(GF_IP, f'GammaFactory_IP{i}')



num_turns=int(2*1e8)
n_part=len(particles_old.x)


lin_tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)

delta_old = particles_old.delta




x=[]
px=[]

a1=0.0020
a2=0.0025

de=-hw0*0.5*2.0*gamma

cutoff=500
    



delta_list = []

for i in tqdm(range(num_turns)):
    x.append(particles0.x[0])
    px.append(particles0.px[0])
    
    if min(a1,a2) < particles0.x[0] < max(a1,a2):
        #print(particles0.delta[0])
        delta_list.append(particles0.delta[0])
        if particles0.delta[0]>0:
            #print('true')
            #particles0.delta+= -0.1*delta_old
            particles0.add_to_energy(-de)
    
    lin_tracker.track(particles0)
    
    
    #plt.scatter(x[i:i+cutoff:],px[i:i+cutoff],color='red')
    #plt.savefig('images/turn'+str(i)+'.png')


#x2=lin_tracker.record_last_track.x  
    
# plt.figure()
# plt.scatter(x[:cutoff:],px[:cutoff],label='first 100 turns')
# plt.scatter(x[-cutoff:],px[-cutoff:],label='last 100 turns')    

# plt.axvline(a1,color='red',label='kick location')
# plt.axvline(a2,color='red')


# plt.legend()
# plt.title('Strong kick with dispersion')
# plt.xlabel('x')
# plt.ylabel('px')

#%%


np.save('cache/x.npy', x)
np.save('cache/px.npy', px)


#%%


# from matplotlib.animation import ArtistAnimation

# fig = plt.figure() 
# ax  = fig.add_subplot(111) 


# images = []

# for i in range(0,num_turns-sample):
#     value=i
#     line, = ax.plot(x[0:sample],px[0:sample])
#     line, = ax.plot(x[int(value):int(value)+sample],px[int(value):int(value)+sample],color='red')
#     images.append((line,))

# line_anim = ArtistAnimation(fig, images, interval=sample, blit=True)
# #line_anim.save('my_animation.mp4')
# line_anim.save('my_animation.gif')
# #plt.show()


#%%

# import mpl_interactions.ipyplot as iplt

# freq = tau = np.linspace(1, num_turns-sample)

# def f_x(freq):
#     return x[int(freq):int(freq)+sample]


# def f_y(freq):
#     return px[int(freq):int(freq)+sample]


# fig, ax = plt.subplots()
# controls = iplt.plot(
#     f_x,
#     f_y,
#     freq=freq,
#     xlim="auto",
#     ylim="auto",
#     label="interactive!",
# )



#%%


# import holoviews as hv


# dim_x  = hv.Dimension('x',  unit='mm', range=(-4,+4))
# dim_px = hv.Dimension('px', unit='mrad', label="x'", range=(-0.12,+0.12))

# #hv.Scatter(x,px,kdims=['x','px'])

# hv.Points((x, px), kdims=[dim_x,dim_px])