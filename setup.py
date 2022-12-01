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
fname_sequence ='/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'



with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xt.Line.from_dict(input_data['line'])


fname_sequence ='/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'



with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
sequence = xt.Line.from_dict(input_data['line'])



#%%
##################
# Build TrackJob #
##################

n_part = int(1e3)

SPS_tracker = xt.Tracker(_context=context, _buffer=buf, line=sequence)


# Build a reference particle
particle_sample = xp.Particles(mass0=m_ion, q0=Z-Ne, p0c=p0c)

sigma_z = 22.5e-2
nemitt_x = 2e-6
nemitt_y = 2.5e-6


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

sequence.particle_ref = particle_sample
twiss = SPS_tracker.twiss(symplectify=True)


del twiss['particle_on_co']
del twiss['_ebe_fields']

#%%
###################
# Linear Transfer #
###################

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


import pickle

with open('cache/twiss.pkl', 'wb') as f:
    pickle.dump(twiss, f)

with open('cache/SPS_lin.json', 'w') as fid:
    json.dump(SPS_lin.to_dict(), fid, cls=xo.JEncoder)

with open('cache/particles_old.json', 'w') as fid:
    json.dump(particles_old.to_dict(), fid, cls=xo.JEncoder)

