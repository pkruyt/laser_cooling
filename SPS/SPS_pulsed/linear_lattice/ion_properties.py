# ion_properties.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants 
from scipy.optimize import curve_fit

clight=constants.speed_of_light
gamma0=190.99019102

class Ion:
    def __init__(self, name, A, Z, q0, excited_lifetime, hw0):
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

        N_pb = int(1.9 * 1e8)  # ion-bunch intensity for lead
        self.N_a = int(N_pb * (self.Z / 82) ** -1.9)  # ion-bunch intensity for arbitrary ion with charge Z
        self.Intensity = self.N_a
        self.gamma_proton = 480.67 #https://acc-models.web.cern.ch/acc-models/sps/2021/scenarios/lhc_proton/
        self.gamma_rel = (self.Z/self.A )*self.gamma_proton
        self.beta_rel = np.sqrt(1-1/(self.gamma_rel*self.gamma_rel)) # ion beta

calcium = Ion(name="Ca$^{17+}$", A=40, Z=20, q0=17, excited_lifetime=0.4279*1e-12, hw0=661.89)
xenon = Ion(name="Xe$^{39+}$", A=129, Z=54, q0=39, excited_lifetime=3*1e-12, hw0=492.22)
lead = Ion(name="Pb$^{79+}$", A=208, Z=82, q0=79, excited_lifetime=76.6e-12, hw0=230.823)

ions=[lead,xenon,calcium]

