# ion_properties.py

import numpy as np
from scipy import constants 

clight=constants.speed_of_light
#gamma0=190.99019102

class Ion:
    def __init__(self, name, A, Z, q0, excited_lifetime, hw0,
                 lambda_l,bunch_length,bunch_intensity,laser_x,gamma_rel,
                 gamma_cooling,gamma_heating,):
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
        #self.Intensity = self.N_a
        self.gamma_rel = gamma_rel
        self.gamma_cooling = gamma_cooling
        self.gamma_heating = gamma_heating
        self.laser_x=laser_x
        self.beta_rel = np.sqrt(1-1/(self.gamma_rel*self.gamma_rel)) # ion beta
        self.lambda_l=lambda_l
        self.bunch_length=bunch_length
        self.bunch_intensity=bunch_intensity

#cooling =(1+4e-4)   heating =(1-4e-4) laser shift
lead = Ion(name="Pb$^{79+}$", A=208, Z=82, q0=79, excited_lifetime=76.6e-12, hw0=230.823,
           lambda_l=1031.8*1e-9,laser_x=-1.2416107382550332*1e-3, bunch_length=0.063, bunch_intensity = 0.9*1e8,
           gamma_rel=96.088235121299187,gamma_cooling=96.07863342564447,gamma_heating=96.11705172635483)


calcium = Ion(name="Ca$^{17+}$", A=40, Z=20, q0=17, excited_lifetime=0.4279*1e-12, hw0=661.89,
              lambda_l=768*1e-9,laser_x=-1.577181208053691*1e-3, bunch_length=0.010, bunch_intensity = 4*1e9,
              gamma_rel=205.08913416882473,gamma_cooling=205.0686404689884,gamma_heating=205.150639852357,
              )

xenon = Ion(name="Xe$^{51+}$", A=129, Z=54, q0=51, excited_lifetime=3*1e-12, hw0=492.22,
            lambda_l=768*1e-9,laser_x=-1.3758389261744963*1e-3, bunch_length=0.082, bunch_intensity = 2*1e8,
            gamma_rel=152.5162392853479,gamma_cooling=152.500998975125,gamma_heating=152.56197849812983
            )

xenon2nd = Ion(name="Xe$^{51+}$ 2nd harmonic", A=129, Z=54, q0=51, excited_lifetime=3*1e-12, hw0=492.22,
               lambda_l=515*1e-9,laser_x=-1.3758389261744963*1e-3, bunch_length=0.082, bunch_intensity = 0.9*1e8,
                gamma_rel=102.25793057356498,gamma_cooling=102.26303967732991,gamma_heating=102.30393089392821,
              ) #need to find optimal laser_x for xenon2nd
                #need to check xenon intensity
