import pickle
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.widgets import Slider



# with open('stuff.pkl', 'rb') as f:
#    x = pickle.load(f)
#    px = pickle.load(f)

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
excited=np.load('cache/excited.npy')

# x0=x[0,:]
# x1=x[1,:]
# x00=x[:,0]

ex0=excited[0,:]
ex00=excited[:,0]

number_of_particles=len(x[:,0])
number_of_turns=len(x[0,:])



#%%


#######
#  X  #
#######

# sample=1000


# # Create the figure and the line that we will manipulate
# fig, ax = plt.subplots()
# line = ax.scatter(x[:,0:sample], px[:,0:sample],color='orange',label='initial')

# ylim_init = ax.get_ylim()
# xlim_init = ax.get_xlim()

# fig.suptitle('Energy reduction with dispersion')
# ax.set_xlabel('x(m)')
# ax.set_ylabel('px')

# ax.legend()

# fig.set_size_inches(18.5, 10.5, forward=True)    
# fig.subplots_adjust(left=0.25, bottom=0.25)

# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Turns',
#     valmin=0,
#     valmax=number_of_turns-sample,
#     valinit=1,
# )

# # The function to be called anytime a slider's value changes
# # def update(val):
# #     value=freq_slider.val
    
# #     line.set_xdata(x[value:500])
# #     line.set_ydata(px[value.val:500])
# #     fig.canvas.draw_idle()


# def update(val):
#     value=freq_slider.val
#     ax.clear()
    
#     #a1=lower_bound_list[int(value)]
#     #a2=max_x_list[int(value)]
    
    
#     ax.scatter(x[:,0:sample], px[:,0:sample],color='orange',label='initial')
#     ax.scatter(x[:,int(value):int(value)+sample],px[:,int(value):int(value)+sample]
#                ,label='turn evolution')
    
#     ax.set_xlabel('x(m)')
#     ax.set_ylabel('px')
    
#     ax.set_xlim(xlim_init)
#     ax.set_ylim(ylim_init)
    
#     ax.legend()


#     #plt.draw()

    
# freq_slider.on_changed(update)

#%%

#######
#  Y  #
#######

sample=1000


# # Create the figure and the line that we will manipulate
# fig, ax = plt.subplots()
# line = ax.scatter(y[:,0:sample], py[:,0:sample],color='orange',label='initial')

# ylim_init = ax.get_ylim()
# xlim_init = ax.get_xlim()

# fig.suptitle('Energy reduction with dispersion')
# ax.set_xlabel('y(m)')
# ax.set_ylabel('py')

# ax.legend()

# fig.set_size_inches(18.5, 10.5, forward=True)    
# fig.subplots_adjust(left=0.25, bottom=0.25)

# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Turns',
#     valmin=0,
#     valmax=number_of_turns-sample,
#     valinit=1,
# )

# # The function to be called anytime a slider's value changes
# # def update(val):
# #     value=freq_slider.val
    
# #     line.set_xdata(x[value:500])
# #     line.set_ydata(px[value.val:500])
# #     fig.canvas.draw_idle()


# def update(val):
#     value=freq_slider.val
#     ax.clear()
    
#     #a1=lower_bound_list[int(value)]
#     #a2=max_x_list[int(value)]
    
    
#     ax.scatter(y[:,0:sample], py[:,0:sample],color='orange',label='initial')
#     ax.scatter(y[:,int(value):int(value)+sample],py[:,int(value):int(value)+sample]
#                 ,label='turn evolution')
    
#     ax.set_xlabel('y(m)')
#     ax.set_ylabel('py')
    
#     ax.set_xlim(xlim_init)
#     ax.set_ylim(ylim_init)
    
#     ax.legend()


#     #plt.draw()

    
# freq_slider.on_changed(update)

#%%
#######
#  Z  #
#######

# sample=1000


# # Create the figure and the line that we will manipulate
# fig, ax = plt.subplots()
# line = ax.scatter(zeta[:,0:sample], delta[:,0:sample],color='orange',label='initial')

# ylim_init = ax.get_ylim()
# xlim_init = ax.get_xlim()

# fig.suptitle('Energy reduction with dispersion')
# ax.set_xlabel('zeta(m)')
# ax.set_ylabel('delta')

# ax.legend()

# fig.set_size_inches(18.5, 10.5, forward=True)    
# fig.subplots_adjust(left=0.25, bottom=0.25)

# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Turns',
#     valmin=0,
#     valmax=number_of_turns-sample,
#     valinit=1,
# )

# # The function to be called anytime a slider's value changes
# # def update(val):
# #     value=freq_slider.val
    
# #     line.set_xdata(x[value:500])
# #     line.set_ydata(px[value.val:500])
# #     fig.canvas.draw_idle()


# def update(val):
#     value=freq_slider.val
#     ax.clear()
    
#     #a1=lower_bound_list[int(value)]
#     #a2=max_x_list[int(value)]
    
    
#     ax.scatter(zeta[:,0:sample], delta[:,0:sample],color='orange',label='initial')
#     ax.scatter(zeta[:,int(value):int(value)+sample],delta[:,int(value):int(value)+sample]
#                ,label='turn evolution')
    
#     ax.set_xlabel('zeta(m)')
#     ax.set_ylabel('delta')
    
#     ax.set_xlim(xlim_init)
#     ax.set_ylim(ylim_init)
    
#     ax.legend()


#     #plt.draw()

    
# freq_slider.on_changed(update)

#%%
###############
#  Z excited  #
###############

# sample=1000

# z0=excited[:,0]
# z2=zeta[:,0:sample][excited[:,1]]
# # Create the figure and the line that we will manipulate
# fig, ax = plt.subplots()
# line = ax.scatter(zeta[:,0:sample], delta[:,0:sample],color='orange',label='initial')
# line = ax.scatter(zeta[:,0:sample][excited[:,0]], delta[:,0:sample][excited[:,0]],color='red',label='excited')


# ylim_init = ax.get_ylim()
# xlim_init = ax.get_xlim()

# fig.suptitle('Energy reduction with dispersion')
# ax.set_xlabel('zeta(m)')
# ax.set_ylabel('delta')

# ax.legend()

# fig.set_size_inches(18.5, 10.5, forward=True)    
# fig.subplots_adjust(left=0.25, bottom=0.25)

# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Turns',
#     valmin=0,
#     valmax=number_of_turns-sample,
#     valinit=1,
# )

# # The function to be called anytime a slider's value changes
# # def update(val):
# #     value=freq_slider.val
    
# #     line.set_xdata(x[value:500])
# #     line.set_ydata(px[value.val:500])
# #     fig.canvas.draw_idle()


# def update(val):
#     value=freq_slider.val
#     ax.clear()
    
#     #a1=lower_bound_list[int(value)]
#     #a2=max_x_list[int(value)]
    
    
#     #ax.scatter(zeta[:,0:sample], delta[:,0:sample],color='orange',label='initial')
#     ax.scatter(zeta[:,int(value):int(value)+sample],delta[:,int(value):int(value)+sample]
#                ,label='turn evolution')
    
#     ax.scatter( zeta[:,int(value):int(value)+sample][excited[:,int(value)+sample]],
#                delta[:,int(value):int(value)+sample][excited[:,int(value)+sample]]
#                ,label='excited')
    
    
#     ax.set_xlabel('zeta(m)')
#     ax.set_ylabel('delta')
    
#     ax.set_xlim(xlim_init)
#     ax.set_ylim(ylim_init)
    
#     ax.legend()


#     #plt.draw()

    
# freq_slider.on_changed(update)