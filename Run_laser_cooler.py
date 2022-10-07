import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp


####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCpu(omp_num_threads=5)
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

buf = context.new_buffer()


#num_turns = int(1e2)
n_part = int(1e0)


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

sigma_dp = sigma_z / beta
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

#%%
##################
# Build TrackJob #
##################

SPS_tracker = xt.Tracker(_context=context, _buffer=buf, line=sequence)


# Build a reference particle
particle_sample = xp.Particles(mass0=m_ion, q0=Z-Ne, p0c=p0c)

particles0 = xp.generate_matched_gaussian_bunch(
         num_particles=n_part,
         #total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         #R_matrix=r_matrix,
         particle_ref=particle_sample,
         tracker=SPS_tracker
         #,steps_r_matrix=steps_r_matrix
         )

particles_old=particles0.copy()
particles00=particles_old.copy()
particles000=particles_old.copy()

sequence.particle_ref = particle_sample
twiss = SPS_tracker.twiss(symplectify=True)



del twiss['particle_on_co']

import pickle

with open('twiss.pkl', 'wb') as f:
    pickle.dump(twiss, f)

num_turns = 1


#%%

SPS_tracker.track(particles0, num_turns=num_turns, turn_by_turn_monitor=True)


x0=SPS_tracker.record_last_track.x
px0=SPS_tracker.record_last_track.px
y0=SPS_tracker.record_last_track.y
py0=SPS_tracker.record_last_track.py
z0=SPS_tracker.record_last_track.zeta
delta0=SPS_tracker.record_last_track.delta

#%%
###################
# Linear Transfer #
###################


arc=xt.LinearTransferMatrix(Q_x=twiss['qx'], Q_y=twiss['qy'],
beta_x_0=twiss['betx'][0], beta_x_1=twiss['betx'][-1], beta_y_0=twiss['bety'][0], beta_y_1=twiss['bety'][-1],
alpha_x_0=twiss['alfx'][0], alpha_x_1=twiss['alfx'][-1], alpha_y_0=twiss['alfy'][0], alpha_y_1=twiss['alfy'][-1],
disp_x_0=twiss['dx'][0], disp_x_1=twiss['dx'][-1], disp_y_0=twiss['dy'][0], disp_y_1=twiss['dy'][-1],
Q_s=twiss['qs'], beta_s=twiss['betz0'],
chroma_x=twiss['dqx'], chroma_y=twiss['dqy'])


SPS_lin = xt.Line()

SPS_lin.append_element(arc,'SPS_LinearTransferMatrix')


lin_tracker = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin)

lin_tracker.track(particles00, num_turns=num_turns, turn_by_turn_monitor=True)

#%%


with open('SPS_lin.json', 'w') as fid:
    json.dump(SPS_lin.to_dict(), fid, cls=xo.JEncoder)

with open('particles_old.json', 'w') as fid:
    json.dump(particles_old.to_dict(), fid, cls=xo.JEncoder)



#%%


# with open('SPS_lin.json', 'r') as fid:
#     loaded_dct = json.load(fid)
# SPS_lin2 = xt.Line.from_dict(loaded_dct)


# # Load particles from json file to selected context
# with open('particles_old.json', 'r') as fid:
#     particles_old2= xp.Particles.from_dict(json.load(fid), _context=context)


# lin_tracker2 = xt.Tracker(_context=context, _buffer=buf, line=SPS_lin2)

# lin_tracker2.track(particles000, num_turns=num_turns, turn_by_turn_monitor=True)


# #%%


# x_lin=lin_tracker.record_last_track.x
# px_lin=lin_tracker.record_last_track.px
# y_lin=lin_tracker.record_last_track.y
# py_lin=lin_tracker.record_last_track.py
# z_lin=lin_tracker.record_last_track.zeta
# delta_lin=lin_tracker.record_last_track.delta

# x_lin2=lin_tracker2.record_last_track.x
# px_lin2=lin_tracker2.record_last_track.px
# y_lin2=lin_tracker2.record_last_track.y
# py_lin2=lin_tracker2.record_last_track.py
# z_lin2=lin_tracker2.record_last_track.zeta
# delta_lin2=lin_tracker2.record_last_track.delta