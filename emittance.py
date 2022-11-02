import numpy as np
import matplotlib.pyplot as plt

def emittance_2d(tracker,particles,disp_x_0):
    
    
    
    x=tracker.record_last_track.x
    px=tracker.record_last_track.px
    
    delta=tracker.record_last_track.delta
    
    x=x-disp_x_0*delta
    
    num_turns=len(x[0,:])
    
    gamma=particles.gamma0[0]
    beta=particles.beta0[0]
    
    cov_list=[]
    for i in range(num_turns):
        x0=x[:,i]
        px0=px[:,i]
    
    
        cov00=np.cov(x0,px0)
    
        det00 = (np.sqrt((np.linalg.det(cov00)))*beta*gamma)
        cov_list.append(det00)
        
   
    #print(cov_list)
    
    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    plt.plot(cov_list)
    
    plt.title('Horizontal emmitance vs turns (2d)')
    plt.ylabel('emittance (m)')
    plt.xlabel('number of turns')    
    return det00
    
def emittance_6d(tracker,particles):
    
    x=tracker.record_last_track.x
    px=tracker.record_last_track.px
    y=tracker.record_last_track.y
    py=tracker.record_last_track.py
    zeta=tracker.record_last_track.zeta
    delta=tracker.record_last_track.delta
    
    
    num_turns=len(x[0,:])
    
    gamma=particles.gamma0
    beta=particles.beta0
    
    
    
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
    
        det00 = np.sqrt((np.linalg.det(cov00)))*beta*gamma
        cov_list.append(det00)
        
   
    
    plt.figure()
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    plt.plot(cov_list)
    
    plt.title('Total emmitance vs turns (6d)')
    plt.ylabel('emittance $(m^3)$')
    plt.xlabel('number of turns')  




