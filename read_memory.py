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

with open('cache/particles_old.json', 'r') as fid:
    particles0= xp.Particles.from_dict(json.load(fid), _context=context)




x_list=[particles0.x]
px_list=[particles0.px]

y_list=[particles0.y]
py_list=[particles0.py]

zeta_list=[particles0.zeta]
delta_list=[particles0.delta]

state_list=[particles0.state]

num_cycles=int(1e3)
#num_cycles=int(1e2)

for i in range(num_cycles):
    globals()[f'x{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/x{i}.npy')
    x_list.append(globals()[f'x{i}'])
    globals()[f'px{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/px{i}.npy')
    px_list.append(globals()[f'px{i}'])
    
    globals()[f'y{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/y{i}.npy')
    y_list.append(globals()[f'y{i}'])
    globals()[f'py{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/py{i}.npy')
    py_list.append(globals()[f'py{i}'])
    
    globals()[f'zeta{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/zeta{i}.npy')
    zeta_list.append(globals()[f'zeta{i}'])
    globals()[f'delta{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/delta{i}.npy')
    delta_list.append(globals()[f'delta{i}'])
    
    globals()[f'state{i}']=np.load(f'/home/pkruyt/Documents/cache_memory/state{i}.npy')
    state_list.append(globals()[f'state{i}'])
    
    
    
x=np.transpose(np.array(x_list))
px=np.transpose(np.array(px_list))
    
y=np.transpose(np.array(y_list))
py=np.transpose(np.array(py_list))
   
zeta=np.transpose(np.array(zeta_list))
delta=np.transpose(np.array(delta_list))

state=np.transpose(np.array(state_list))
    
    
np.save('cache_memory/x.npy', x)
np.save('cache_memory/px.npy', px)
np.save('cache_memory/y.npy', y)
np.save('cache_memory/py.npy', py)
np.save('cache_memory/zeta.npy', zeta)
np.save('cache_memory/delta.npy', delta)
np.save('cache_memory/state.npy', state)     
  

import os
import glob

files = glob.glob('/home/pkruyt/Documents/cache_memory/*')
for f in files:
    os.remove(f)
