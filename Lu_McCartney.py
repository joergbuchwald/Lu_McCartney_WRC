from math import exp, pow
from scipy.special import erf
import numpy as np


class LuMcCartney:
    """Lu and McCartney (2024) soil water retention model for bentonite-based buffer materials.
    References:
    https://link.springer.com/article/10.1007/s11440-024-02341-9
    https://www.sciencedirect.com/science/article/pii/S0266352X25006858#b0050
    """
    def __init__(self):
        self.R = 8.314 #J/(mol·K)
        self.hp = 6.626e-34 #J·s
        self.kB = 1.381e-23 #J/K
        # parameters for adsorptive water content
        self.nu_w = 1.8e-5 #m^3/mol
        self.E1_EL = 7534 #J/mol
        self.zeta_s = 2.8 #meq/g
        self.bw = 0.18 #g/g
        self.CEC_max = 0.776 #meq/g
        self.A = 0.92
        self.B = 0.9
        self.T_a = 373.15 #K
        self.T_b = 1273.15 #K
        # parameters for capillary water content
        self.SSA = 7e5 #m^2/kg
        self.AH = 10213 #kPa
        self.Delta_H = -0.516 #J/m^2
        self.epsilon_s = 3.5
        self.n_s = 1.55
        self.n_w = 1.33
        self.alpha_0_inv = 3.3 #MPa
        self.eta_alpha = 5.0
        self.C_1 = -0.00151
        self.n = 1.15

        # material properties
        self.phi = 0.45 #m3/m3
        self.theta_s = self.phi #m3/m3
    
    def water_content(self, psi, T):
        theta_ad = self.adsorptive_water_content(psi, T)
        theta_cap = self.capillary_water_content(psi, T)
        return theta_ad + theta_cap

    def adsorptive_water_content(self, psi, T):
        theta_ad = self.theta_a_max(T) * (1 - exp((psi-self.psi_max(T))/psi)**self.M)
        return theta_ad

    def capillary_water_content(self, psi, T):
        theta_cap = (self.theta_s - self.adsorptive_water_content(psi, T))/2 *(1 - erf(np.sqrt(2)*psi/self.phi_c(T)*((self.chi(T)+T)/(chi_r+Tr))))
        return theta_cap

    def theta_a_max(self, T):
        #eq. 6
        pass
    
    def psi_max(self, T):
        #eq. 4
        pass
    
    def c(self):
        # eq. 5
        pass

    def CEC(self, T):
        # eq. 8
        pass

    def psi_c(self, T):
        # eq. 25
        pass

    def A_H(self):
        # eq23
        pass

    def chi(self, T):
        # eq. 18
        pass

    def delta_h(self):
        # eq 13
        pass

    def alpha(self):
        # eq. 26
        pass


    