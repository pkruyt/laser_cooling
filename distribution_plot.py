import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.widgets import Slider
from tqdm import tqdm

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


# x=np.load('cache_memory/x.npy')
# px=np.load('cache_memory/px.npy')
# y=np.load('cache_memory/y.npy')
# py=np.load('cache_memory/py.npy')
# zeta=np.load('cache_memory/zeta.npy')
# delta=np.load('cache_memory/delta.npy')
# state=np.load('cache_memory/state.npy')


num_turns=np.shape(x)[1]


x1=x[:,1]
x2=x[:,2]
z2=zeta[:,2]
 

#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# xlim_init=(-0.005865734476107509, 0.0054770649623092985)
# ylim_init=(-0.00012998686719493268, 0.00012136987865081761)


# xlim_init=(-0.000865734476107509, 0.0008770649623092985)
# ylim_init=(-0.00000998686719493268, 0.00000836987865081761)

 
for turn in tqdm(range(num_turns)):
    if turn % 10 ==0:
        x1 = zeta[:,turn]
        y1 = delta[:,turn]
    
        # x1=np.expand_dims(x1,axis=1)
        # y1=np.expand_dims(y1,axis=1)
    
    
        x_exc = x1[state[:,turn]==2]
        y_exc = y1[state[:,turn]==2]
    
    
        fraction=len(x_exc)/len(x1)
    
        #fontsize=12
    
        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    
        ax_main.scatter(x1,y1,marker='.',label='all particles',linewidths=5)    
        ax_main.scatter(x_exc,y_exc,marker='.',label='excited',linewidths=5)
        #ax_main.set(xlabel="x(mm)", ylabel="px")
        ax_main.set_xlabel('z')
        ax_main.set_ylabel('delta')
    
    
        ax_xDist.hist(x_exc,bins=100,align='mid')
        ax_xDist.set(ylabel='count')
        ax_xCumDist = ax_xDist.twinx()
        ax_xCumDist.hist(x_exc,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid')
        ax_xCumDist.tick_params('y', colors='r')
        ax_xCumDist.set_ylabel('cumulative',color='r')
    
        ax_yDist.hist(y_exc,bins=100,orientation='horizontal',align='mid')
        ax_yDist.set(xlabel='count')
        ax_yCumDist = ax_yDist.twiny()
        ax_yCumDist.hist(y_exc,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid',orientation='horizontal')
        ax_yCumDist.tick_params('x', colors='r')
        ax_yCumDist.set_xlabel('cumulative',color='r')
    
    
        # ax_main.axvline(laser_x*1e3, color='red',label='laser location')
        ax_main.legend(loc='best')
        #ax_main.figtext(0.5,0.5,'fraction of excited ions:'+str(fraction))
        # ax_main.set_xlim(xlim_init)
        # ax_main.set_ylim(ylim_init)
    
        
    
    
        fig.text(0.25,0.15,"%.2f%% of ions excited" % (100*fraction),fontsize=15)
        fig.suptitle('Fraction of particles that are excited',x=0.5,y=0.92,fontsize=20)
        #plt.show()
    
        fig.savefig(f'images/temp{turn}.png')
        plt.close(fig)

