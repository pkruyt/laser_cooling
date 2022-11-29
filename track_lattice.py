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






# Load particles from json file to selected context
with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

num_particles=len(particles0.x)    


# particles0.y=0
# particles0.py=0


#SPS=xt.Line(sequence)



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


num_turns=int(1e5)

tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)



monitor = xt.ParticlesMonitor(_context=context,
                              start_at_turn=0, stop_at_turn=num_turns,
                              #n_repetitions=3,      # <--
                              #repetition_period=20, # <--
                              num_particles=num_particles)


for iturn in tqdm(range(num_turns)):
    monitor.track(particles0)
    tracker.track(particles0)
   
    
x=monitor.x
px=monitor.px
y=monitor.y
py=monitor.py
zeta=monitor.zeta
delta=monitor.delta
state=monitor.state


np.save('cache/x.npy', x)
np.save('cache/px.npy', px)
np.save('cache/y.npy', y)
np.save('cache/py.npy', py)
np.save('cache/zeta.npy', zeta)
np.save('cache/delta.npy', delta)
np.save('cache/state.npy', state)   
 

#%%
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

 
# for turn in tqdm(range(num_turns)):

#     x1 = zeta[:,turn]
#     y1 = delta[:,turn]

#     x1=np.expand_dims(x1,axis=1)
#     y1=np.expand_dims(y1,axis=1)


#     x = x1[state[:,turn]==2]
#     y = y1[state[:,turn]==2]





#     fraction=len(x)/len(x1)

#     #fontsize=12

#     fig = plt.figure(figsize=(12,12))
#     gs = gridspec.GridSpec(3, 3)
#     ax_main = plt.subplot(gs[1:3, :2])
#     ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
#     ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

#     ax_main.scatter(x1,y1,marker='.',label='all particles',linewidths=5)    
#     ax_main.scatter(x,y,marker='.',label='excited',linewidths=5)
#     #ax_main.set(xlabel="x(mm)", ylabel="px")
#     ax_main.set_xlabel('z')
#     ax_main.set_ylabel('delta')


#     ax_xDist.hist(x,bins=100,align='mid')
#     ax_xDist.set(ylabel='count')
#     ax_xCumDist = ax_xDist.twinx()
#     ax_xCumDist.hist(x,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid')
#     ax_xCumDist.tick_params('y', colors='r')
#     ax_xCumDist.set_ylabel('cumulative',color='r')

#     ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
#     ax_yDist.set(xlabel='count')
#     ax_yCumDist = ax_yDist.twiny()
#     ax_yCumDist.hist(y,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid',orientation='horizontal')
#     ax_yCumDist.tick_params('x', colors='r')
#     ax_yCumDist.set_xlabel('cumulative',color='r')


#     # ax_main.axvline(laser_x*1e3, color='red',label='laser location')
#     ax_main.legend(loc='best')
#     #ax_main.figtext(0.5,0.5,'fraction of excited ions:'+str(fraction))





#     fig.text(0.25,0.15,"%.2f%% of ions excited" % (100*len(x)/len(x1)),fontsize=15)
#     fig.suptitle('Fraction of particles that are excited',x=0.5,y=0.92,fontsize=20)
#     #plt.show()

#     fig.savefig(f'images/temp{turn}.png')
#     plt.close(fig)

