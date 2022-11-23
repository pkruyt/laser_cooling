import numpy as np 

# Ion properties:
m_u = 931.49410242e6 # eV/c^2 -- atomic mass unit
A = 207.98 # Lead-208
Z = 82  # Number of protons in the ion (Lead)
Ne = 3 # Number of remaining electrons (Lithium-like)
m_e = 0.511e6 # eV/c^2 -- electron mass
m_p = 938.272088e6 # eV/c^2 -- proton mass
c = 299792458.0 # m/s


m_ion = A*m_u + Ne*m_e # eV/c^2

equiv_proton_momentum = 236e9 # eV/c = gamma_p*m_p*v

gamma_p = np.sqrt( 1 + (equiv_proton_momentum/m_p)**2 ) # equvalent gamma for protons in the ring
sigma_dp = 2e-4 # relative ion momentum spread

p0c = equiv_proton_momentum*(Z-Ne) # eV/c
gamma = np.sqrt( 1 + (p0c/m_ion)**2 ) # ion relativistic factor
beta = np.sqrt(1-1/(gamma*gamma)) # ion beta



def gamma_p(p):
    return np.sqrt( 1 + (p/m_ion)**2 ) # equvalent gamma for protons in the ring



gamma_low=gamma_p(p0c-sigma_dp )
gamma_mid=gamma_p(p0c          )
gamma_high=gamma_p(p0c+sigma_dp)


diff=gamma_high-gamma_low