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


# x=np.load('cache_memory/x.npy')
# px=np.load('cache_memory/px.npy')
# y=np.load('cache_memory/y.npy')
# py=np.load('cache_memory/py.npy')
# zeta=np.load('cache_memory/zeta.npy')
# delta=np.load('cache_memory/delta.npy')
# state=np.load('cache_memory/state.npy')



number_of_particles=len(x[:,0])
number_of_turns=len(x[0,:])


#%%

#######
#  X  #
#######

# ylim_manual=(-0.00011957501236985238, 0.00011119590367708776)
# xlim_manual=(-0.005120480059662477, 0.005930270032610736)




x=zeta
px=delta

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line = ax.scatter(x[:,0], px[:,0],color='orange',label='initial')

ylim_init = ax.get_ylim()
xlim_init = ax.get_xlim()

fig.suptitle('Energy reduction with dispersion')
ax.set_xlabel('x(m)')
ax.set_ylabel('px')

ax.legend()

fig.set_size_inches(18.5, 10.5, forward=True)    
fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Turns',
    valmin=0,
    valmax=number_of_turns,
    valinit=0,
)

# The function to be called anytime a slider's value changes
# def update(val):
#     value=freq_slider.val
    
#     line.set_xdata(x[value:500])
#     line.set_ydata(px[value.val:500])
#     fig.canvas.draw_idle()





def update(val):
    turn=freq_slider.val
    ax.clear()
    
    turn=int(turn)
    x1 = x[:,turn]
    y1 = px[:,turn]
    
    
    
    # x1=np.expand_dims(x1,axis=1)
    # y1=np.expand_dims(y1,axis=1)


    x_exc = x1[state[:,turn]==2]
    y_exc = y1[state[:,turn]==2]

    
    
    #a1=lower_bound_list[int(value)]
    #a2=max_x_list[int(value)]
    
    # x_exc = x[state[:,turn]==2]
    # px_exc = px[state[:,turn]==2]
    
    ax.scatter(x[:,0], px[:,0],color='orange',label='initial')
    ax.scatter(x[:,turn],px[:,turn]
                ,label='turn evolution')
    
    
    ax.scatter( x_exc,
                y_exc
                ,label='excited',color='red')
    
    
    ax.set_xlabel('x(m)')
    ax.set_ylabel('px')
    
    ax.set_xlim(xlim_init)
    ax.set_ylim(ylim_init)
    
    # ax.set_xlim(xlim_manual)
    # ax.set_ylim(ylim_manual)
    
    ax.legend()


    #plt.draw()

    
freq_slider.on_changed(update)

