import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.widgets import Slider


#%%
def emittance_2d(x,px):
    
    #number_of_particles=len(x[:,0])
    #number_of_turns=len(x[0,:])
    
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
        
        cov_list2=cov_list/cov_list[0]
   
    #print(cov_list)
    
    # plt.figure()
    # ax = plt.gca()
    # ax.ticklabel_format(useOffset=False)
    
    # plt.plot(cov_list)
    
    # plt.title('Horizontal emmitance vs turns (2d)')
    # plt.ylabel('emittance (m)')
    # plt.xlabel('number of turns')    
    return cov_list2


