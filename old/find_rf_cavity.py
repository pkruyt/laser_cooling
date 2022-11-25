import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp

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

   
line=input_data['line']['elements']
line_names=input_data['line']['element_names']

# Find dictionary matching value in list
res = []

for sub in line:
    if sub['__class__'] == "Cavity":
        res.append(sub)
    
cavity_index = line.index(res[0])        
        
#%%
#find skew quad
skew=[]

for sub in line:
    if sub['__class__'] == "Multipole" and  sub['order'] == 2 and sub['ksl']==[0,0,0]:   
   
        skew.append(sub)
     



skew2= [s for s in line_names if 'q' in s]
     


#%%
##############
# Read Twiss #
##############

with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)

mass0=particles0.mass0
q0=particles0.q0
p0c=particles0.p0c[0]


particle_sample = xp.Particles(mass0=mass0, q0=q0, p0c=p0c)



a_file = open("cache/twiss.pkl", "rb")

twiss = pickle.load(a_file)   

#remove int values from dict
for key in twiss.copy():
   
    if type(twiss[key])==np.ndarray:
        print(type(twiss[key]))
    if type(twiss[key])!=np.ndarray:
        del twiss[key]










#%%
# #line1
# line_elements1 = line_elements[:cavity_index]
# line_names1=line_names[:cavity_index]


# line1=dict()

# line1['elements'] = line_elements1
# line1['element_names'] = line_names1

# sequence1 = xt.Line.from_dict(line1)

# #%%
# #line 2
# line_elements2 = line_elements[cavity_index+1:]
# line_names2=line_names[cavity_index+1:]


# line2=dict()

# line2['elements'] = line_elements2
# line2['element_names'] = line_names2

# sequence2 = xt.Line.from_dict(line1)

# #%%



# SPS_tracker1 = xt.Tracker(_context=context, _buffer=buf, line=sequence1)        
# SPS_tracker2 = xt.Tracker(_context=context, _buffer=buf, line=sequence2)        


# # SPS_tracker1.track(particles0, num_turns=100,
# #                turn_by_turn_monitor=True)



# #twiss1 = SPS_tracker1.twiss(method='4d',particle_ref = particle_sample)
# twiss2 = SPS_tracker2.twiss(method='4d',particle_ref = particle_sample)




# # del twiss1['particle_on_co']
# # #del twiss2['particle_on_co']

# # with open('cache/twiss1.pkl', 'wb') as f:
# #     pickle.dump(twiss1, f)
# # # with open('cache/twiss2.pkl', 'wb') as f:
# # #     pickle.dump(twiss2, f)



# print('done')