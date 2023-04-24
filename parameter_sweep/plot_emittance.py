import glob
import numpy as np
import matplotlib.pyplot as plt


# variable_name='laser_x'
#variable_name='laser_wavelength'
# variable_name='sigma_t'
# variable_name='laser_waist_radius'
variable_name='laser_energy'
variable_name='theta'

path=f'/home/pkruyt/Documents/sweep_{variable_name}/emittance_results/'

#emitt_z

numpy_vars = {}

for file in glob.glob(path +'emitt_z/' '*.npy'):
    print(file)
    numpy_vars[file] = np.load(file)
    
final_z = []    
# plt.figure(figsize=(18.5, 10.5))    
plt.figure()
for key in numpy_vars:
        position1=key.find(':')
        position2=key.find('.n')
        label=key[position1+1:position2]
        
        print(label)
        plt.plot(numpy_vars[key],label=label)
        final_z.append(numpy_vars[key][-1])

   
        
plt.legend()        
plt.xlabel('Number of turns')
plt.ylabel('Fraction of initial emittance')
plt.title('emitt_z : ' + variable_name)

plt.savefig(f'/home/pkruyt/cernbox/PLOTS/parameter_sweep_results/emitt_z/emitt_z:{variable_name}')



numpy_vars = {}

for file in glob.glob(path +'emitt_x/' '*.npy'):
    print(file)
    numpy_vars[file] = np.load(file)
    
    
# plt.figure(figsize=(18.5, 10.5))    
plt.figure() 
for key in numpy_vars:
        position1=key.find(':')
        position2=key.find('.n')
        label=key[position1+1:position2]
        
        print(label)
        plt.plot(numpy_vars[key],label=label)
        
        
plt.legend()   
plt.title('emitt_x : ' + variable_name)
plt.xlabel('Number of turns')
plt.ylabel('Fraction of initial emittance')

plt.savefig(f'/home/pkruyt/cernbox/PLOTS/parameter_sweep_results/emitt_x/emitt_x:{variable_name}')