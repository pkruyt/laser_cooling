import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp
from tqdm import tqdm
from scipy import constants 
import xfields as xf

line = xt.Line.from_json('sps.json')
particle_ref=line.particle_ref

line.build_tracker()
twiss=line.twiss()

clight=constants.speed_of_light
circumference = line.get_length()

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

for ion in [lead]:

    # Ion properties:
    q0 = ion.q0
    mass0 = ion.mass0

    gamma = ion.gamma_cooling
    beta= ion.beta_rel
    p0c = mass0*gamma*beta #eV/c

    bunch_intensity = ion.bunch_intensity

    particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=q0, gamma=gamma)

    line.particle_ref=particle_ref

    nemitt = 1.5e-6 # m*rad (normalized emittance)
    sigma_z = 0.063 # m
    sigma_z = ion.bunch_length


    emittance=nemitt/(beta*gamma)

    num_particles=int(1e3)

    particles = xp.generate_matched_gaussian_bunch(
            num_particles=num_particles,
            total_intensity_particles=bunch_intensity,
            nemitt_x=nemitt, nemitt_y=nemitt, sigma_z=sigma_z,
            particle_ref=particle_ref,
            line=line,        
            )

    particles0=particles.copy()
    # sigma_dp=2e-4  
    sigma_dp=np.std(particles.delta)

    ##################
    # Laser Cooler #
    ##################

    #laser-ion beam collision angle
    theta_l = 2.6*np.pi/180 # rad
    nx = 0; ny = -np.sin(theta_l); nz = -np.cos(theta_l)

    # Ion excitation energy:
    ion_excited_lifetime=ion.excited_lifetime
    hw0 = ion.hw0 # eV
    hc=constants.hbar*clight/constants.e # eV*m (ħc)
    lambda_0 = 2*np.pi*hc/hw0 # m -- ion excitation wavelength

    lambda_l = lambda_0*gamma*(1 + beta*np.cos(theta_l)) # m -- laser wavelength

    # Shift laser wavelength for fast longitudinal cooling:
    #lambda_l = lambda_l*(1+1*sigma_dp) # m

    lambda_l = ion.lambda_l

    laser_frequency = clight/lambda_l # Hz
    sigma_w = 2*np.pi*laser_frequency*sigma_dp
    #sigma_w = 2*np.pi*laser_frequency*sigma_dp/2 # for fast longitudinal cooling

    sigma_t = 1/sigma_w # sec -- Fourier-limited laser pulse
    print('Laser pulse duration sigma_t = %.2f ps' % (sigma_t/1e-12))
    print('Laser wavelength = %.2f nm' % (lambda_l/1e-9))

    laser_waist_radius = 1.3e-3 #m
    laser_energy = 5e-3

    laser_x = ion.laser_x

    GF_IP = xt.PulsedLaser(
                    laser_x=laser_x,
                    laser_y=0,
                    laser_z=0,
                    
                    laser_direction_nx = 0,
                    laser_direction_ny = ny,
                    laser_direction_nz = nz,
                    laser_energy         = laser_energy, # J
                    laser_duration_sigma = sigma_t, # sec
                    laser_wavelength = lambda_l, # m
                    laser_waist_radius = laser_waist_radius, # m
                    laser_waist_shift = 0, # m
                    ion_excitation_energy = hw0, # eV
                    ion_excited_lifetime  = ion_excited_lifetime, # sec                   
                    )

    # simulation parameters: simulate 10 s of cooling, and take data once every 100 ms
    max_time_s = 100
    int_time_s = 0.01
    T_per_turn = circumference/(clight*beta)
    num_turns = int(max_time_s/T_per_turn)
    save_interval = int(int_time_s/T_per_turn)

    # create a monitor object, to reduce holded data
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                                n_repetitions=int(num_turns/save_interval),
                                repetition_period=save_interval,
                                num_particles=num_particles)

    ibs_kick = xf.IBSKineticKick(num_slices=50)
    line.configure_intrabeam_scattering(
    element=ibs_kick, name="ibskick", index=-1, update_every=50                                       )


    line.discard_tracker()
    IP_index=16675   
    line.insert_element('monitor', element=monitor, index=IP_index)
    line.insert_element('GF_IP', element=GF_IP, index=IP_index) #this way monitor comes after the laser

    # at interaction points: #from https://anaconda.org/petrenko/li_like_ca_in_sps/notebook
    beta_x  =  twiss.betx[IP_index]
    beta_y  =  twiss.bety[IP_index]
    alpha_x =  twiss.alfx[IP_index]
    alpha_y =  twiss.alfy[IP_index]

    gamma_x=twiss.gamx[IP_index]
    gamma_y=twiss.gamy[IP_index]

    Dx  =  twiss.dx[IP_index]
    Dpx =  twiss.dpx[IP_index]

    Dy  =  twiss.dy[IP_index]
    Dpy =  twiss.dpy[IP_index]

    particles=particles0.copy()

    #context = xo.ContextCpu()
    context = xo.ContextCupy()
    line.build_tracker(_context=context)
    #line.optimize_for_tracking()
   

    line.track(particles, num_turns=num_turns,
                turn_by_turn_monitor=False,with_progress=True)

    # extract relevant values
    x = monitor.x[:,:,0]
    px = monitor.px[:,:,0]
    y = monitor.y[:,:,0]
    py = monitor.py[:,:,0]
    delta = monitor.delta[:,:,0]
    zeta = monitor.zeta[:,:,0]
    state = monitor.state[:,:,0]
    time = monitor.at_turn[:, 0, 0] * T_per_turn

    action_x = (gamma_x*(x-Dx*delta)**2 + 2*alpha_x*(x-Dx*delta)*(px-Dpx*delta)+ beta_x*(px-Dpx*delta)**2)
    action_y = (gamma_y*(y-Dy*delta)**2 + 2*alpha_y*(y-Dy*delta)*(py-Dpy*delta)+ beta_y*(py-Dpy*delta)**2)

    emittance_x_twiss=np.mean(action_x,axis=1)*gamma/2

    np.savez(f'./{ion.name}.npz', x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
            action_x=action_x,action_y=action_y,emittance_x=emittance_x_twiss,
            state=state, time=time,s_per_turn=T_per_turn)
