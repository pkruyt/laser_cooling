import pickle
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.widgets import Slider



# with open('stuff.pkl', 'rb') as f:
#    x = pickle.load(f)
#    px = pickle.load(f)

# with open('stuff2.pkl', 'rb') as f:
#    x2,px2 = pickle.load(f)
   
x=np.load('cache/x.npy').tolist()
px=np.load('cache/px.npy').tolist()
max_x_list=np.load('cache/max_x_list.npy').tolist()
min_x_list=np.load('cache/min_x_list.npy').tolist()
lower_bound_list=np.load('cache/lower_bound_list.npy').tolist()

# x=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/x.npy')
# px=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/px.npy')
# max_x_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/max_x_list.npy')
# min_x_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/min_x_list.npy')
# lower_bound_list=np.load('cache/good_results/cooling_moving_laser_non_linear(100x)/lower_bound_list.npy')



num_turns = np.size(x)


a1=-0.0010
a2=-0.0008

a1=0.0015
a2=0.0032


#%%
##################
#      Slider    #
##################

sample=500

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line = ax.scatter(x[0:sample], px[0:sample],color='orange',label='initial')

fig.suptitle('Energy reduction with dispersion')
ax.set_xlabel('x(m)')
ax.set_ylabel('px')
ax.axvline(x=a1,color='red',label='cooling zone')
ax.axvline(x=a2,color='red')
ax.legend()
    
fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Turns',
    valmin=0,
    valmax=num_turns-sample,
    valinit=1,
)

# The function to be called anytime a slider's value changes
# def update(val):
#     value=freq_slider.val
    
#     line.set_xdata(x[value:500])
#     line.set_ydata(px[value.val:500])
#     fig.canvas.draw_idle()


def update(val):
    value=freq_slider.val
    ax.clear()
    
    #a1=lower_bound_list[int(value)]
    #a2=max_x_list[int(value)]
    
    ax.axvline(x=a1,color='red',label='cooling zone')
    ax.axvline(x=a2,color='red')
    ax.scatter(x[0:sample], px[0:sample],color='orange',label='initial')
    ax.scatter(x[int(value):int(value)+sample],px[int(value):int(value)+sample]
               ,label='turn evolution')
    ax.legend()


    #plt.draw()

    
freq_slider.on_changed(update)



#%%
##################
#    Make gif    #
##################

# from tqdm import tqdm
# import os
# import glob

# files = glob.glob('images/*')
# for f in files:
#     os.remove(f)
    
    

# num_figs=100

# interval=num_turns/num_figs


# fig_number=0

# for i in tqdm(range(num_turns)):
    
            
#     if i % interval == 0:
#         fig_number+=1
#         #plt.figure();
#         plt.title(f'Energy reduction with dispersion. Turn {i}')
#         plt.xlabel('x(m)')
#         plt.ylabel('px')
        
#         #plt.xlim([min(min_x_list),max(max_x_list)])
#         a1=lower_bound_list[int(i)]
#         a2=max_x_list[int(i)] 
        
#         plt.axvline(x=a1,color='red',label='cooling zone')
#         plt.axvline(x=a2)
#         plt.scatter(x[0:sample:],px[0:sample],color='orange',label='Initial')
#         plt.scatter(x[i:i+sample:],px[i:i+sample],color='blue',label='turn evolution')
        
#         plt.legend()
#         plt.savefig('images/turn'+str(fig_number)+'.png',dpi=80)
#         plt.clf()


#%%





