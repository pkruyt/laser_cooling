import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.widgets import Slider


# with open('stuff2.pkl', 'rb') as f:
#    x2,px2 = pickle.load(f)
   
# x=np.load('cache/x.npy')
# px=np.load('cache/px.npy')
# max_x_list=np.load('cache/max_x_list.npy')
# min_x_list=np.load('cache/min_x_list.npy')
# lower_bound_list=np.load('cache/lower_bound_list.npy')

# x=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/x.npy')
# px=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/px.npy')
# max_x_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/max_x_list.npy')
# min_x_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/min_x_list.npy')
# lower_bound_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/lower_bound_list.npy')



# x=np.load('cache/coupling/x.npy')
# px=np.load('cache/coupling/px.npy')
# y=np.load('cache/coupling/y.npy')
# py=np.load('cache/coupling/py.npy')
# zeta=np.load('cache/coupling/zeta.npy')
# delta=np.load('cache/coupling/delta.npy')

x=np.load('cache/x.npy')
px=np.load('cache/px.npy')
y=np.load('cache/y.npy')
py=np.load('cache/py.npy')
zeta=np.load('cache/zeta.npy')
delta=np.load('cache/delta.npy')
state=np.load('cache/state.npy')


x=np.load('cache2/x.npy')
px=np.load('cache2/px.npy')
y=np.load('cache2/y.npy')
py=np.load('cache2/py.npy')
zeta=np.load('cache2/zeta.npy')
delta=np.load('cache2/delta.npy')
state=np.load('cache2/state.npy')


x0=x[0,:]
x1=x[1,:]
x00=x[:,0]

ex0=state[0,:]
ex00=state[:,0]

number_of_particles=len(x[:,0])
number_of_turns=len(x[0,:])


#%%

import json
import numpy as np
import xtrack as xt
import xpart as xp
import sys
sys.path.append("../statistical_emittance/statisticalEmittance/")
from statisticalEmittance import *


bunch_intensity = 1e11
sigma_z = 22.5e-2
n_part = int(5e5)
nemitt_x = 2e-6
nemitt_y = 2.5e-6

filename = ('/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
with open(filename, 'r') as fid:
    ddd = json.load(fid)
tracker = xt.Tracker(line=xt.Line.from_dict(ddd['line']))
part_ref = xp.Particles.from_dict(ddd['particle'])
tracker.line.particle_ref = part_ref

particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=part_ref,
         tracker=tracker)


r=statisticalEmittance(particles)
epsn_x = []
epsn_y = []
for ii in range(5):
    print(ii)
    tracker.track(particles)
    r.setInputDistribution(particles)
    epsn_x.append(r.getNormalizedEmittanceX())
    epsn_y.append(r.getNormalizedEmittanceY())

print('epsn_x = ',epsn_x)
print('epsn_y = ',epsn_y)