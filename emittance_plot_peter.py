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
def emittance_2d(x,px):
    
    
    
    #x=tracker.record_last_track.x
    #px=tracker.record_last_track.px
    
    #delta=tracker.record_last_track.delta
    
    #x=x-disp_x_0*delta
    
    num_turns=len(x[0,:])
    
    #gamma=particles.gamma0[0]
    #beta=particles.beta0[0]
    
    cov_list=[]
    for i in range(num_turns):
        x0=x[:,i]
        px0=px[:,i]
    
    
        cov00=np.cov(x0,px0)
    
        #det00 = (np.sqrt((np.linalg.det(cov00)))*beta*gamma)
        det00 = (np.sqrt((np.linalg.det(cov00))))
        cov_list.append(det00)
        
   
    #print(cov_list)
    
    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    plt.plot(cov_list)
    
    plt.title('Horizontal emmitance vs turns (2d)')
    plt.ylabel('emittance (m)')
    plt.xlabel('number of turns')    
    return 


def emittance_6d():
    
    # x=tracker.record_last_track.x
    # px=tracker.record_last_track.px
    # y=tracker.record_last_track.y
    # py=tracker.record_last_track.py
    # zeta=tracker.record_last_track.zeta
    # delta=tracker.record_last_track.delta
    
    
    num_turns=len(x[0,:])
    
    # gamma=particles.gamma0
    # beta=particles.beta0
    
    
    
    cov_list=[]
    for i in range(num_turns):
        x0=x[:,i]
        px0=px[:,i]
                      
        y0=y[:,i]
        py0=py[:,i]
               
        zeta0=zeta[:,i]
        pdelta0=delta[:,i]
    
        
        
        data = np.array([x0,px0,
                         y0,py0,
                         zeta0,pdelta0])
        
    
        cov00=np.cov(data)
    
        #det00 = np.sqrt((np.linalg.det(cov00)))*beta*gamma
        det00 = np.sqrt((np.linalg.det(cov00)))
        cov_list.append(det00)
        
   
    
    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    plt.plot(cov_list)
    
    plt.title('Total emmitance vs turns (6d)')
    plt.ylabel('emittance $(m^3)$')
    plt.xlabel('number of turns')  

emittance_2d(y,py)
