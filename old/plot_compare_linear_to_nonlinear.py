import numpy as np
import matplotlib.pyplot as plt


x_lin=np.load('cache/comparison/x_lin.npy')
px_lin=np.load('cache/comparison/px_lin.npy')
y_lin=np.load('cache/comparison/y_lin.npy')
py_lin=np.load('cache/comparison/py_lin.npy')
zeta_lin=np.load('cache/comparison/zeta_lin.npy')
delta_lin=np.load('cache/comparison/delta_lin.npy')

x_nonlin=np.load('cache/comparison/x_nonlin.npy')
px_nonlin=np.load('cache/comparison/px_nonlin.npy')
y_nonlin=np.load('cache/comparison/y_nonlin.npy')
py_nonlin=np.load('cache/comparison/py_nonlin.npy')
zeta_nonlin=np.load('cache/comparison/zeta_nonlin.npy')
delta_nonlin=np.load('cache/comparison/delta_nonlin.npy')


#%%


from scipy import fftpack


def get_fourier_freq(x):

    #x=zeta_nonlin.flatten()
    x=x.flatten()
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) 
    
    fig, ax = plt.subplots()
    
    ax.plot(freqs.flatten(), np.abs(X))
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    # ax.set_xlim(-f_s / 2, f_s / 2)
    # ax.set_ylim(-5, 110)
    
    
    max_index=np.argmax(np.abs(X))
    max_freq = freqs[max_index]
    return max_freq

fourier_qx=get_fourier_freq(x_nonlin)
fourier_qy=get_fourier_freq(y_nonlin)
fourier_qs=get_fourier_freq(zeta_nonlin)