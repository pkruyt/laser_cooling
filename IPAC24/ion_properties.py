# ion_properties.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants 
from scipy.optimize import curve_fit

clight=constants.speed_of_light
gamma0=190.99019102

class Ion:
    def __init__(self, name, A, Z, q0, excited_lifetime, hw0,gamma_rel,
                 gamma_cooling,gamma_heating,lambda_l,laser_x):
        m_u = 931.49410242e6  # eV/c^2 -- atomic mass unit
        m_e = 0.511e6  # eV/c^2 -- electron mass
        m_p = 938.272088e6  # eV/c^2 -- proton mass

        self.name = name
        self.A = A
        self.Z = Z
        self.q0 = q0  # e
        self.excited_lifetime = excited_lifetime  # s
        self.hw0 = hw0  # eV

        self.ne = Z - q0
        self.mass0 = self.A * m_u + self.ne * m_e  # eV/c^2

        N_pb = int(0.9 * 1e8)  # ion-bunch intensity for lead
        self.N_a = int(N_pb * (self.Z / 82) ** -1.9)  # ion-bunch intensity for arbitrary ion with charge Z
        self.Intensity = self.N_a
        self.gamma_rel = gamma_rel
        self.gamma_cooling = gamma_cooling
        self.gamma_heating = gamma_heating
        self.laser_x=laser_x
        self.beta_rel = np.sqrt(1-1/(self.gamma_rel*self.gamma_rel)) # ion beta
        self.lambda_l=lambda_l

lead = Ion(name="Pb$^{79+}$", A=208, Z=82, q0=79, excited_lifetime=76.6e-12, hw0=230.823,
           gamma_rel=96.088235121299187,gamma_cooling=96.07863342564447,gamma_heating=96.11705172635483,
           lambda_l=1031.8*1e-9,laser_x=-1.2416107382550332*1e-3)





