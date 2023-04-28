#%%

import numpy as np
import random
import json    
import os  
import math


class LinearTransferMatrix(Element):
    _description = [
        ("nx", "", "Laser direction x", 0.0),
        ("ny", "", "Laser direction y", 0.0),
        ("nz", "", "Laser direction z", -1.0),
        ("laser_x", "m", "Laser position x", 0.0),
        ("laser_y", "m", "Laser position y", 0.0),
        ("laser_z", "m", "Laser position z", 0.0),
        
        ("laser_waist_shift", "m", "laser waist shift", 0.0),
        ("laser_waist_radius", "m", "laser waist radius", 1e-3),
       
        ("laser_energy", "J", "laser_energy", 0.0),
        ("laser_sigma_t", "s", "laser duration sigma", 1e-12),
        
        ("laser_wavelength", "m", "laser_wavelength", 1034.0e-9),
        ("ion_excitation_energy", "eV", "ion_excitation_energy", 68.6e3),
        ("ion_excitation_g1", "s", "ion_excitation_g1", 2),
        ("ion_excitation_g2", "s", "ion_excitation_g2", 2),
        ("ion_excited_lifetime", "s", "ion_excited_lifetime", 3.9e-17),
    ]
    


    def track(self,p):
        #retrieve particles parameters
        p0c = p.p0c*(1.0+p.delta)  # eV
        m0=p.mass0
        x=p.x
        y=p.y
        z=p.zeta
        state=p.state
        
        #constants
        c = 299792458.0#; // m/s
        hbar = 1.054571817e-34#; // J*sec
        eV = 1.602176634e-19#; // J
        
        #retrieve laser cooler parameters
        nx=self.nx
        ny=self.ny
        nz=self.nz
        
        laser_x=self.laser_x
        laser_y=self.laser_y
        laser_z=self.laser_z
        w0=self.laser_waist_radius
        laser_waist_shift=self.laser_waist_shift
        
        laser_energy=self.laser_energy
        laser_wavelength=self.laser_wavelength
        laser_sigma_t=self.laser_sigma_t
        
        ion_excitation_energy=self.ion_excitation_energy
        ion_excited_lifetime=self.ion_excited_lifetime
        
        #compute derived parameters
        laser_Rayleigh_length = np.pi*w0*w0/laser_wavelength
        I0 = np.sqrt(2/np.pi)*(laser_energy/laser_sigma_t)/(np.pi*w0*w0)#; // W/m^2
        OmegaTransition = ion_excitation_energy*eV/hbar#; // rad/sec
        
        # Map of Excitation:
        fname = os.joinpath('/home/pkruyt/cernbox/xsuite/xtrack/xtrack/beam_elements/IonLaserIP_data/map_of_excitation.json')
        with open(fname, 'r') as f:
            map_data = json.load(f)
            self.Excitation = np.array(map_data['Excitation probability'])
            self.N_OmegaRabiTau_values, self.N_DeltaDetuningTau_values = np.shape(self.Excitation)
            self.OmegaRabiTau_max = map_data['OmegaRabi*tau_pulse max']
            self.DeltaDetuningTau_max  = map_data['Delta_detuning*tau_pulse max']
            self.Map_of_Excitation = self.Excitation.flatten()
        
        DeltaDetuningTau_max=self.DeltaDetuningTau_max
        OmegaRabiTau_max=self.OmegaRabiTau_max
            
        dDeltaDetuningTau = DeltaDetuningTau_max/(self.N_DeltaDetuningTau_values-1.0)
        dOmegaRabiTau = OmegaRabiTau_max/(self.N_OmegaRabiTau_values-1.0)
    
    
        
        #start
        gamma = np.sqrt(1.0 + p0c*p0c/(m0*m0))
        beta = np.sqrt(1.0 - 1.0/(gamma*gamma))
        beta_x = p*p0c/p.m0/gamma
        beta_y = p*p0c/m0/gamma
        beta_z = np.sqrt(beta*beta - beta_x*beta_x -beta_y*beta_y)
        
        vx = c*beta_x  # m/sec
        vy = c*beta_y  # m/sec
        vz = c*beta_z  # m/sec
        
        # Collision of ion with the laser pulse:
        # The position of the laser beam center is rl=rl0+ct*n. We can find the moment
        # when a particle with a position r=r0+vt collides with the laser as the moment
        # when r−rl is perpendicular to n. Then (r−rl,n)=0, which yields the equation
        # (r0,n)+(v,n)t−(rl0,n)−ct(n,n)=0. Hence
        # tcol=(r0−rl0,n)/[c−(v,n)]
        
        tcol = ( (x-laser_x)*nx + (y-laser_y)*ny + (z-laser_z)*nz ) / (c - (vx*nx+vy*ny+vz*nz))  # sec
        
        xcol = x + vx*tcol  # m
        ycol = y + vy*tcol  # m
        zcol = z + vz*tcol  # m
        
        # r^2 to the laser center = |r-rl| at the moment tcol:
        r2 = (
        (xcol - (laser_x + c * nx * tcol)) ** 2 +
        (ycol - (laser_y + c * ny * tcol)) ** 2 +
        (zcol - (laser_z + c * nz * tcol)) ** 2
        )
        
        Z_to_laser_focus = laser_waist_shift - tcol * c
        
        w = w0 * np.sqrt(1.0 + (Z_to_laser_focus / laser_Rayleigh_length) ** 2)
        I = 4.0 * gamma * gamma * I0 * (w0 / w) * (w0 / w) * np.exp(-2.0 * r2 / (w * w))
        
        OmegaRabi = (hbar * c / (ion_excitation_energy * eV)) * np.sqrt(I * 2 * np.pi / (ion_excitation_energy * eV * ion_excited_lifetime))
        
        OmegaRabiTau = OmegaRabi * laser_sigma_t / (2.0 * gamma)
        
        cos_theta = -(nx * vx + ny * vy + nz * vz) / (beta * c)
        laser_omega_ion_frame = (2.0 * np.pi * c / laser_wavelength) * (1.0 + beta * cos_theta) * gamma
        
        
        DeltaDetuningTau = abs(
            (OmegaTransition - laser_omega_ion_frame)*laser_sigma_t/(2.0*gamma)
        )
        
        if state > 0:
            if DeltaDetuningTau < DeltaDetuningTau_max and OmegaRabiTau > OmegaRabiTau_max:
                # In case of a very high laser field:
                p.set_state(2) # Excited particle
            elif DeltaDetuningTau < DeltaDetuningTau_max and OmegaRabiTau > dOmegaRabiTau/10.0:
                row = int(math.floor(OmegaRabiTau/dOmegaRabiTau))
                col = int(math.floor(DeltaDetuningTau/dDeltaDetuningTau))
                idx = row*self.N_DeltaDetuningTau_values + col
                excitation_probability = self.Map_of_Excitation[idx]
                rnd = random.random()
                if rnd < excitation_probability:
                    p.set_state(2) # Excited particle
                    # photon recoil (from emitted photon!):
                    rnd = random.random()
                    p.add_to_energy(-ion_excitation_energy * rnd * 200000.0 * gamma, 0)
            else:
                # Set the particle state to 1 (still)
                p.set_state(1)
    

